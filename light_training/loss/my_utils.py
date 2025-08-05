import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from skimage import feature
import random
from math import exp


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


class PercepHook:
    '''
    Pytorch forward hook for computing the perceptual loss
    without modifying the original VGG16 network
    '''
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()


class Percep_loss(nn.Module):
    '''
    compute perceptual loss between fused image and input image
    '''
    def __init__(self, vgg, block_idx, device):
        '''
        block_index: the index of the block in VGG16 network, int or list
        int represents single layer perceptual loss
        list represents multiple layers perceptual loss
        '''
        super(Percep_loss, self).__init__()
        self.block_idx = block_idx
        self.device = device
        # load vgg16_bn model features
        self.vgg = vgg.features.to(device).eval()
        #self.loss = nn.MSELoss()

        # unable gradient update
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # remove maxpooling layer and relu layer
        # TODO:check this part on whether we want relu or not
        bns = [i - 2 for i, m in enumerate(self.vgg) if isinstance(m, nn.MaxPool2d)]

        # register forward hook
        self.hooks = [PercepHook(self.vgg[bns[i]]) for i in block_idx]
        self.features = self.vgg[0: bns[block_idx[-1]] + 1]

    def forward(self, inputs, targets):
        '''
        compute perceptual loss between inputs and targets
        '''
        if inputs.shape[1] == 1:
            # expand 1 channel image to 3 channel, [B, 1, H, W] -> [B, 3, H, W]
            inputs = inputs.expand(-1, 3, -1, -1)
        if targets.shape[1] == 1:
            targets = targets.expand(-1, 3, -1, -1)
        
        # get vgg output
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        assert len(input_features) == len(target_features), 'number of input features and target features should be the same'
        loss = 0
        for i in range(len(input_features)):
            #loss += self.loss(input_features[i], target_features[i]) # mse loss
            loss += ((input_features[i] - target_features[i]) ** 2).mean() # l2 norm
        
        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


class SSIM(nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def patch_mean(images, patch_shape):
    """
    Computes the local mean of an image or set of images.

    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)

    Returns:
        Tensor same size as the image, with local means computed independently for each channel.

    Example::
        >>> images = torch.randn(4, 3, 15, 15)           # 4 images, 3 channels, 15x15 pixels each
        >>> patch_shape = 3, 5, 5                        # 3 channels, 5x5 pixels neighborhood
        >>> means = patch_mean(images, patch_shape)
        >>> expected_mean = images[3, 2, :5, :5].mean()  # mean of the third image, channel 2, top left 5x5 patch
        >>> computed_mean = means[3, 2, 5//2, 5//2]      # computed mean whose 5x5 neighborhood covers same patch
        >>> computed_mean.isclose(expected_mean).item()
        1
    """
    channels, *patch_size = patch_shape
    dimensions = len(patch_size)
    padding = tuple(side // 2 for side in patch_size)

    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    # Convolution with these weights will effectively compute the channel-wise means
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.to(images.device)

    # Make convolution operate on single channels
    channel_selector = torch.eye(channels).byte()
    weights[1 - channel_selector] = 0

    result = conv(images, weights, padding=padding, bias=None)

    return result


def patch_std(image, patch_shape):
    """
    Computes the local standard deviations of an image or set of images.

    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)

    Returns:
        Tensor same size as the image, with local standard deviations computed independently for each channel.

    Example::
        >>> images = torch.randn(4, 3, 15, 15)           # 4 images, 3 channels, 15x15 pixels each
        >>> patch_shape = 3, 5, 5                        # 3 channels, 5x5 pixels neighborhood
        >>> stds = patch_std(images, patch_shape)
        >>> patch = images[3, 2, :5, :5]
        >>> expected_std = patch.std(unbiased=False)     # standard deviation of the third image, channel 2, top left 5x5 patch
        >>> computed_std = stds[3, 2, 5//2, 5//2]        # computed standard deviation whose 5x5 neighborhood covers same patch
        >>> computed_std.isclose(expected_std).item()
        1
    """
    return (patch_mean(image**2, patch_shape) - patch_mean(image, patch_shape)**2).sqrt()


def channel_normalize(template):
    """
    Z-normalize image channels independently.
    """
    reshaped_template = template.clone().view(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False))

    return reshaped_template.view_as(template)


class NCC(torch.nn.Module):
    """
    Computes the [Zero-Normalized Cross-Correlation][1] between an image and a template.

    Example:
        >>> lena_path = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
        >>> lena_tensor = torch.Tensor(plt.imread(lena_path)).permute(2, 0, 1).cuda()
        >>> patch_center = 275, 275
        >>> y1, y2 = patch_center[0] - 25, patch_center[0] + 25
        >>> x1, x2 = patch_center[1] - 25, patch_center[1] + 25
        >>> lena_patch = lena_tensor[:, y1:y2 + 1, x1:x2 + 1]
        >>> ncc = NCC(lena_patch)
        >>> ncc_response = ncc(lena_tensor[None, ...])
        >>> ncc_response.max()
        tensor(1.0000, device='cuda:0')
        >>> np.unravel_index(ncc_response.argmax(), lena_tensor.shape)
        (0, 275, 275)

    [1]: https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
    """
    def __init__(self, template, keep_channels=False):
        super().__init__()

        self.keep_channels = keep_channels

        channels, *template_shape = template.shape
        dimensions = len(template_shape)
        self.padding = tuple(side // 2 for side in template_shape)

        self.conv_f = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]
        self.normalized_template = channel_normalize(template)
        ones = template.dim() * (1, )
        self.normalized_template = self.normalized_template.repeat(channels, *ones)
        # Make convolution operate on single channels
        channel_selector = torch.eye(channels).byte()
        self.normalized_template[1 - channel_selector] = 0
        # Reweight so that output is averaged
        patch_elements = torch.Tensor(template_shape).prod().item()
        self.normalized_template.div_(patch_elements)

    def forward(self, image):
        result = self.conv_f(image, self.normalized_template, padding=self.padding, bias=None)
        std = patch_std(image, self.normalized_template.shape[1:])
        result.div_(std)
        if not self.keep_channels:
            result = result.mean(dim=1)

        return result