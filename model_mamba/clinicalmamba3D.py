import math
import torch
import numpy as np
from torch.autograd import forward_ad
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ModuleNotFoundError:
    from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


######################## 3D ##########################################

class stem3D(nn.Module):
    '''
    3D stem module, downsample
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1):
        super().__init__()
        # add norm before?
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        x = self.conv(x)
        return x


class DGC3D(nn.Module):
    '''
    DGC, Dilated Gated Convolutional Network
    '''
    def __init__(self, in_channels, out_channels, d_rate):
      super().__init__()
      self.norm1 = nn.BatchNorm3d(in_channels)
      self.norm2 = nn.BatchNorm3d(out_channels)
      self.norm3 = nn.BatchNorm3d(out_channels)
      self.conv_proj1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")
      self.conv_proj2 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding="same")
      self.conv_expand = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding="same", dilation=d_rate)
      self.act1 = nn.LeakyReLU(0.2, inplace=True)
      self.act2 = nn.LeakyReLU(0.2, inplace=True)
      self.act3 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
      resid = x
      x_1 = self.act1(self.conv_proj1(self.norm1(x)))
      # print("x_ref", x_ref.shape)
      x_2 = self.act2(self.conv_proj2(self.norm2(x)))
      # print("x_proj1", x_proj1.shape)
      x_w1 = x_1 * x_2
      x_proj2 = self.act3(self.conv_expand(self.norm3(x_w1)))
      return x_proj2 + resid


class SingleMambaBlock3D(nn.Module):
    def __init__(self, dim, H, W, Z):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=2, d_state=8, bimamba_type='3dd3', 
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W, input_z=Z)

    def forward(self, input):
        # print("run single mamba!")
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        output = self.block(input)
        # output = self.norm1(output)
        # print("run finished")
        return output + skip


class CrossMambaBlock3D(nn.Module):
    def __init__(self, dim, H, W, Z):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=2, d_state=8, bimamba_type='3dd3_fuse', 
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W, input_z=Z)
    def forward(self, input0, input1):
        # print("run cross mamba!")
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        output = self.block(input0, extra_emb=input1)
        # output = self.norm2(output)
        # print("run finished")
        return output + skip


class CrossChannelExcitation3D(nn.Module):
    def __init__(self, in_channel, out_channel, se_ratio=16, mode='mamba', spe_channels=None):
        super(CrossChannelExcitation3D, self).__init__()

        self.mode = mode
        self.maxpooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.avgpooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        # if mode == 'mamba':
        #     self.block = nn.Sequential(
        #         nn.Linear(1, channels),
        #         nn.LayerNorm(channels),
        #         Mamba(channels, expand=1, d_state=8, bimamba_type='v2', if_devide_out=True, use_norm=True),
        #         nn.Linear(channels, 1)
        #     )
        # else:
        #     self.block = nn.Sequential(
        #         nn.Conv2d(spe_channels, spe_channels // se_ratio, 1, 1, 0, bias=False),
        #         nn.LeakyReLU(),
        #         nn.Conv2d(spe_channels // se_ratio, spe_channels, 1, 1, 0, bias=False),
        #     )
        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.LeakyReLU(),

            nn.Conv3d(spe_channels // se_ratio, spe_channels, 1, 1, 0, bias=False),
        )
        self.convin = nn.Conv3d(in_channel, out_channel, 1, 1, 0, bias=False)
        self.convin2 = nn.Conv3d(in_channel, out_channel, 1, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, ref):
        # input: (B, C, H, W) | ref: (B, C, H, W)
        input = self.convin(input)
        ref = self.convin2(ref)
        max_input = self.maxpooling(input)
        avg_input = self.avgpooling(input)
        # max_input: (B, C, 1, 1) | avg_input: (B, C, 1, 1)
        added = max_input + avg_input
        # added: (B, C, 1, 1)
        added = self.sigmoid(added)
        exited = ref * added
        # exited: (B, C, H, W)
        # if self.mode == 'mamba':
        #     output = self.block(input.squeeze(-1)).unsqueeze(-1)
        # else:
        #     output = self.block(input)
        return exited
    

class MambaEnc3D(nn.Module):
    def __init__(self, dim, H, W, Z, depth = 1):
        super().__init__()
        self.conv_in = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.downsample1 = stem3D(32, 32, (2,2,2), stride=2, padding=0)
        self.dgc1 = DGC3D(32, 32, d_rate=1) #DGC(64, 64)
        self.downsample2 = stem3D(32, 64, (2,2,2), stride=2, padding=0)
        self.dgc2 = DGC3D(64, 64, d_rate=3) #DGC(128, 128)
        self.downsample3 = stem3D(64, 128, (2,2,2), stride=2, padding=0)
        self.dgc3 = DGC3D(128, 128, d_rate=5) #DGC(128, 128)

        self.x_mamba_layers = nn.ModuleList([])
        self.y_mamba_layers = nn.ModuleList([])
        for _ in range(depth):
            self.x_mamba_layers.append(SingleMambaBlock3D(dim, H, W, Z))
            self.y_mamba_layers.append(SingleMambaBlock3D(dim, H, W, Z))
        self.x_cross_mamba = CrossMambaBlock3D(dim, H, W, Z)
        self.y_cross_mamba = CrossMambaBlock3D(dim, H, W, Z)
        self.out_proj = nn.Linear(dim, dim)
        self.dim = dim
        self.H = H
        self.W = W
        self.Z = Z
        self.instancenorm1 = nn.InstanceNorm3d(128)
        self.instancenorm2 = nn.InstanceNorm3d(128)
        self.layernorm1 = LayerNorm(128)
        # self.channel_mix = ChannelExchange(p=2)
        self.shallow_fusion1 = nn.Conv3d(self.dim*2,self.dim,3,1,1)
        self.shallow_fusion2 = nn.Conv3d(self.dim*2,self.dim,3,1,1)

    def forward(self, x, y):
        Z_orig, H_orig, W_orig = x.shape[2], x.shape[3], x.shape[4]

        conv_inx = self.conv_in(x) # 32, 128, 128, 128
        conv_iny = self.conv_in(y)

        x = self.downsample1(conv_inx) # 32, 64, 64, 64
        # resid_x1 = x
        x = self.dgc1(x)
        # x = x + resid_x1

        y = self.downsample1(conv_iny) # 32, 64, 64, 64
        # resid_y1 = y
        y = self.dgc1(y)
        # y = y + resid_y1
        
        x = self.downsample2(x) # 64, 32, 32, 32
        # resid_x2 = x
        x = self.dgc2(x)
        # x = x + resid_x2
        
        y = self.downsample2(y) # 64, 32, 32, 32
        # resid_y2 = y
        y = self.dgc2(y)
        # y = y + resid_y2

        x = self.downsample3(x) # 128, 16, 16, 16
        # resid_x3 = x
        x = self.dgc3(x)
        # x = x + resid_x3

        y = self.downsample3(y) # 128, 16, 16, 16
        # resid_y3 = y
        y = self.dgc3(y)
        # y = y + resid_y3

        Z, H, W = y.shape[2], y.shape[3], y.shape[4]
        # x, y: BCHW, now B(HW)C
        # permute Z to last dim
        x = x.permute(0,1,3,4,2).contiguous()  # B, C, H, W, Z
        y = y.permute(0,1,3,4,2).contiguous()
        
        x = x.flatten(2,-1).permute(0,2,1).contiguous()
        y = y.flatten(2,-1).permute(0,2,1).contiguous()
        # print(x.shape, y.shape)

        if self.dim != x.shape[-1]:
            raise ValueError("Input size mismatch")

        x_mamba = x
        y_mamba = y

        for x_layer, y_layer in zip(self.x_mamba_layers, self.y_mamba_layers):
            x_mamba = x_layer(x_mamba)
            y_mamba = y_layer(y_mamba)

        # fusion mamba, and channel mix
        # x_mamba, y_mamba, x_mamba_resid, y_mamba_resid = self.channel_mix(x_mamba, y_mamba, x, y)
        # x_mamba, y_mamba, x_mamba_resid, y_mamba_resid = self.channel_mix(x_mamba, y_mamba, x_mamba_resid, y_mamba_resid)
        
        # shallow fusion
        # x_mamba = x_mamba.permute(0, 2, 1).contiguous().view(-1, self.dim, H, W)
        # y_mamba = y_mamba.permute(0, 2, 1).contiguous().view(-1, self.dim, H, W)
        
        # x_mamba = self.shallow_fusion1(torch.cat([x_mamba, y_mamba], dim=1)) + x_mamba
        # y_mamba = self.shallow_fusion2(torch.cat([x_mamba, y_mamba], dim=1)) + y_mamba
        
        # fuse
        x_fuse = self.x_cross_mamba(x_mamba, y_mamba)
        y_fuse = self.y_cross_mamba(y_mamba, x_mamba)
        
    
        
        fusion = self.out_proj((x_fuse + y_fuse) / 2)
        # merge
        # x = x + fusion
        # y = y + fusion
        # x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        # y = rearrange(y, 'b (h w) c -> b c h w', h=H, w=W)
        # x = self.instancenorm1(x)
        # y = self.instancenorm2(y)
        # output = torch.cat([x, y], dim=1)
        output = rearrange(fusion, 'b (h w z) c -> b c h w z', h=H, w=W, z=Z).permute(0,1,4,2,3).contiguous()
        merge = output #x + y + output
        return merge, x_fuse, y_fuse, conv_inx, conv_iny

class Recon3D(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels=channel, out_channels=channel//2, kernel_size=3, stride=1, padding="same")
        # self.group_norm = nn.GroupNorm(channel//2, channel)
        self.instancenorm1 = nn.InstanceNorm3d(channel//2)
        self.recon_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                        nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding="same"),
                                        nn.BatchNorm3d(64),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        #nn.ReLU(),
                                        nn.Upsample(scale_factor=2, mode="nearest"),
                                        nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding="same"),
                                        #nn.ReLU(),
                                        nn.BatchNorm3d(32),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Upsample(scale_factor=2, mode="nearest"),
                                        nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding="same"),
                                        nn.BatchNorm3d(32))
        self.cross_channel_excitation = CrossChannelExcitation3D(in_channel=32, out_channel=32, se_ratio=16, mode='mamba', spe_channels=32)
        self.recon_conv2 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding="same")
        self.final_activation = nn.Sigmoid()
    def forward(self, fused, orig_x, orig_y):
        # x = self.conv_in(x)
        # x = self.instancenorm1(x)
        fused = self.recon_conv(fused)
        crossc1 = self.cross_channel_excitation(orig_x, orig_y)
        fused = fused + crossc1
        crossc2 = self.cross_channel_excitation(orig_y, orig_x)
        fused = fused + crossc2
        x = self.recon_conv2(fused)
        x = self.final_activation(x)
        return x

class MambaFuse3D(nn.Module):
    def __init__(self, dim, H, W, Z, outchannel, depth = 1):
        """
        dim: latent channel
        H, W: latent H,W
        outchannel: channel after fuse
        """
        super().__init__()
        self.mambaenc = MambaEnc3D(dim, H, W, Z, depth)
        self.recon = Recon3D(outchannel)
    
    def forward(self, x, y):
        fuse_merge, x_fuse, y_fuse, orig_x, orig_y = self.mambaenc(x, y)
        recons = self.recon(fuse_merge, orig_x, orig_y)
        return recons, x_fuse, y_fuse