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


def L1_norm(source_en_a, source_en_b):
    result = []
    narry_a = source_en_a
    narry_b = source_en_b

    dimension = source_en_a.shape

    # caculate L1-norm
    temp_abs_a = torch.abs(narry_a)
    temp_abs_b = torch.abs(narry_b)
    _l1_a = torch.sum(temp_abs_a, dim=1)
    _l1_b = torch.sum(temp_abs_b, dim=1)

    _l1_a = torch.sum(_l1_a, dim=0)
    _l1_b = torch.sum(_l1_b, dim=0)
    with torch.no_grad():
        l1_a = _l1_a.detach()
        l1_b = _l1_b.detach()

    # caculate the map for source images
    mask_value = l1_a + l1_b
    # print("mask_value 的size",mask_value.size())

    mask_sign_a = l1_a / mask_value
    mask_sign_b = l1_b / mask_value

    array_MASK_a = mask_sign_a
    array_MASK_b = mask_sign_b
    # print("array_MASK_b 的size",array_MASK_b.size())
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            temp_matrix = array_MASK_a * narry_a[i, j, :, :] + array_MASK_b * narry_b[i, j, :, :]
            # print("temp_matrix 的size",temp_matrix.size())
            result.append(temp_matrix)  

    result = torch.stack(result, dim=-1)

    result = result.reshape((dimension[0], dimension[1], dimension[2], dimension[3]))

    return result


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

class stem2D(nn.Module):
    '''
    2D stem module, downsample
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1):
        super().__init__()
        # add norm before?
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        x = self.conv(x)
        return x


class DGC(nn.Module):
    '''
    DGC, Dilated Gated Convolutional Network
    '''
    def __init__(self, in_channels, out_channels, d_rate):
      super().__init__()
      self.norm1 = nn.BatchNorm2d(in_channels)
      self.norm2 = nn.BatchNorm2d(out_channels)
      self.norm3 = nn.BatchNorm2d(out_channels)
      self.norm4 = nn.BatchNorm2d(out_channels)
      self.conv_proj1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")
      self.conv_proj2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="same")
      self.conv_expand = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same", dilation=d_rate)
      self.conv_pyramid = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")
      self.act1 = nn.LeakyReLU(0.2, inplace=True)
      self.act2 = nn.LeakyReLU(0.2, inplace=True)
      self.act3 = nn.LeakyReLU(0.2, inplace=True)
      self.act4 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
      resid = x
      x_1 = self.act1(self.conv_proj1(self.norm1(x)))
      # print("x_ref", x_ref.shape)
      x_2 = self.act2(self.conv_proj2(self.norm2(x)))
      # print("x_proj1", x_proj1.shape)
      x_w1 = x_1 * x_2
      x_proj2 = self.act3(self.conv_expand(self.norm3(x_w1)))
      x_proj2 = self.act4(self.norm4(x_proj2))
      conv_path = self.conv_pyramid(x_proj2) + self.conv_pyramid(self.conv_pyramid(x_proj2)) + self.conv_pyramid(self.conv_pyramid(self.conv_pyramid(x_proj2)))
      return resid + conv_path


class SingleMambaBlock(nn.Module):
    def __init__(self, dim, H, W, mamba_type = 'v6'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=2, d_state=8, bimamba_type=mamba_type, 
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)

    def forward(self, input):
        # print("run single mamba!")
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        output = self.block(input)
        # output = self.norm1(output)
        # print("run finished")
        return output + skip


class CrossMambaBlock(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=2, d_state=8, bimamba_type='v7', 
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)
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


class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        B, C, N = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((B, -1))
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        out_x1 = out_x1.permute(0, 2, 1)
        out_x2 = out_x2.permute(0, 2, 1)
        return out_x1, out_x2


class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.I1encoder = Mamba(dim,bimamba_type=None)
        self.I2encoder = Mamba(dim,bimamba_type=None)
        self.norm1 = nn.LayerNorm(dim) # LayerNorm(dim,'with_bias')
        self.norm2 = nn.LayerNorm(dim) # LayerNorm(dim,'with_bias')
        self.ChannelExchange = ChannelExchange(p=2)
    def forward(self, I1,I2
                ,I1_residual,I2_residual):

        I1_residual = I1+I1_residual
        I2_residual = I2+I2_residual
        I1 = self.norm1(I1_residual)
        I2 = self.norm2(I2_residual)
        B,N,C = I1.shape
 
        I1_swap, I2_swap = self.ChannelExchange(I1, I2) 

        I1_swap = self.I1encoder(I1_swap)
        I2_swap = self.I2encoder(I2_swap)
        return I1_swap,I2_swap,I1_residual,I2_residual


class CrossChannelExcitation(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CrossChannelExcitation, self).__init__()
        
        self.maxpooling = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.convin = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)
        self.convin2 = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)
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


class MambaEnc(nn.Module):
    def __init__(self, dim, H, W, depth = 1):
        super().__init__()
        self.conv_in = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.downsample1 = stem2D(32, 32, 2, stride=2, padding=0)
        self.dgc1 = DGC(32, 32, d_rate=1) #DGC(64, 64)
        self.downsample2 = stem2D(32, 64, 2, stride=2, padding=0)
        self.dgc2 = DGC(64, 64, d_rate=3) #DGC(128, 128)
        self.downsample3 = stem2D(64, 128, 2, stride=2, padding=0)
        self.dgc3 = DGC(128, 128, d_rate=5) #DGC(128, 128)

        self.x_mamba_layers = nn.ModuleList([])
        self.y_mamba_layers = nn.ModuleList([])
        for _ in range(depth):
            self.x_mamba_layers.append(SingleMambaBlock(dim, H, W))
            self.y_mamba_layers.append(SingleMambaBlock(dim, H, W))
        self.x_cross_mamba = CrossMambaBlock(dim, H, W)
        self.y_cross_mamba = CrossMambaBlock(dim, H, W)
        self.out_proj = nn.Linear(dim, dim)
        self.dim = dim
        self.H = H
        self.W = W
        self.instancenorm1 = nn.InstanceNorm2d(128)
        self.instancenorm2 = nn.InstanceNorm2d(128)
        self.layernorm1 = LayerNorm(128)
        self.channel_mix = TokenSwapMamba(dim)
        self.shallow_fusion1 = nn.Conv2d(self.dim*2,self.dim,3,1,1)
        self.shallow_fusion2 = nn.Conv2d(self.dim*2,self.dim,3,1,1)

    def forward(self, x, y):
        H_orig, W_orig = x.shape[2], x.shape[3]
        
        conv_inx = self.conv_in(x) # 32, 256, 256
        conv_iny = self.conv_in(y)

        # brats, no first downsample, image size is 128
        # x = self.downsample1(conv_inx) # 32, 128,128
        # resid_x1 = x
        x = self.dgc1(conv_inx)
        # x = x + resid_x1

        # y = self.downsample1(conv_iny)
        # resid_y1 = y
        y = self.dgc1(conv_iny)
        # y = y + resid_y1
        
        x = self.downsample2(x) # 64, 64, 64
        # resid_x2 = x
        x = self.dgc2(x)
        # x = x + resid_x2
        
        y = self.downsample2(y)
        # resid_y2 = y
        y = self.dgc2(y)
        # y = y + resid_y2

        x = self.downsample3(x) # 128, 32, 32
        # resid_x3 = x
        x = self.dgc3(x)
        # x = x + resid_x3

        y = self.downsample3(y)
        # resid_y3 = y
        y = self.dgc3(y)
        # y = y + resid_y3

        H, W = y.shape[2], y.shape[3]
        # x, y: BCHW, now B(HW)C
        x = x.flatten(2,3).permute(0,2,1).contiguous()
        y = y.flatten(2,3).permute(0,2,1).contiguous()
        # print(x.shape, y.shape)

        if self.dim != x.shape[-1]:
            raise ValueError("Input size mismatch")

        x_mamba = x
        y_mamba = y

        for x_layer, y_layer in zip(self.x_mamba_layers, self.y_mamba_layers):
            x_mamba = x_layer(x_mamba)
            y_mamba = y_layer(y_mamba)

        # fusion mamba, and channel mix
        x_mamba, y_mamba, x_mamba_resid, y_mamba_resid = self.channel_mix(x_mamba, y_mamba, x, y)
        x_mamba, y_mamba, x_mamba_resid, y_mamba_resid = self.channel_mix(x_mamba, y_mamba, x_mamba_resid, y_mamba_resid)
        # print(f"x_mamba size: {x_mamba.shape}, y_mamba size: {y_mamba.shape} after channel mix")
        # shallow fusion
        # x_mamba = x_mamba.permute(0, 2, 1).contiguous().view(-1, self.dim, H, W)
        # y_mamba = y_mamba.permute(0, 2, 1).contiguous().view(-1, self.dim, H, W)
        
        # x_mamba = self.shallow_fusion1(torch.cat([x_mamba, y_mamba], dim=1)) + x_mamba
        # y_mamba = self.shallow_fusion2(torch.cat([x_mamba, y_mamba], dim=1)) + y_mamba
        
        # fuse
        x_fuse = self.x_cross_mamba(x_mamba, y_mamba)
        y_fuse = self.y_cross_mamba(y_mamba, x_mamba)
        # print(f"x_fuse, y_fuse size, after cross mamba: {x_fuse.shape}, {y_fuse.shape}")
        
        
        fusion = self.out_proj((x_fuse + y_fuse) / 2)
        # merge
        # x = x + fusion
        # y = y + fusion
        # x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        # y = rearrange(y, 'b (h w) c -> b c h w', h=H, w=W)
        # x = self.instancenorm1(x)
        # y = self.instancenorm2(y)
        # output = torch.cat([x, y], dim=1)
        output = fusion # rearrange(fusion, 'b (h w) c -> b c h w', h=H, w=W)
        merge = output #x + y + output
        return merge, x_fuse, y_fuse, conv_inx, conv_iny

class Recon(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels=channel, out_channels=channel//2, kernel_size=3, stride=1, padding="same")
        # self.group_norm = nn.GroupNorm(channel//2, channel)
        self.layernorm = nn.LayerNorm(128)
        self.mambab = nn.Sequential(*[SingleMambaBlock(128, 32, 32, mamba_type=None) for i in range(3)])
        self.recon_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding="same"),
                                        nn.BatchNorm2d(64),
                                        # nn.ReLU(inplace=True),
                                        nn.ReLU(),
                                        nn.Upsample(scale_factor=2, mode="nearest"),
                                        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding="same"),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32))
                                        #nn.LeakyReLU(0.2, inplace=True),
                                        #nn.Upsample(scale_factor=2, mode="nearest"),
                                        #nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding="same"),
                                        #nn.BatchNorm2d(32))
                                        #nn.Sigmoid())
        self.cross_channel_excitation = CrossChannelExcitation(in_channel=32, out_channel=32)
        self.recon_conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding="same")
        self.final_activation = nn.Sigmoid()

    def forward(self, fused, orig_x, orig_y):
        fused = self.mambab(fused)
        # x = self.conv_in(x)
        fused = self.layernorm(fused)
        fused = rearrange(fused, 'b (h w) c -> b c h w', h=32, w=32)
        fused = self.recon_conv(fused)
        crossc1 = self.cross_channel_excitation(orig_x, orig_y)
        fused = fused + crossc1
        crossc2 = self.cross_channel_excitation(orig_y, orig_x)
        fused = fused + crossc2
        x = self.recon_conv2(fused)
        x = self.final_activation(x)
        return x


class MambaFuse(nn.Module):
    def __init__(self, dim, H, W, outchannel, depth = 1):
        """
        dim: latent channel
        H, W: latent H,W
        outchannel: channel after fuse
        """
        super().__init__()
        self.mambaenc = MambaEnc(dim, H, W, depth)
        self.recon = Recon(outchannel)
    
    def forward(self, x, y):
        fuse_merge, x_fuse, y_fuse, orig_x, orig_y = self.mambaenc(x, y)
        recons = self.recon(fuse_merge, orig_x, orig_y)
        return recons, x_fuse, y_fuse
