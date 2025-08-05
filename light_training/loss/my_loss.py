import numpy as np
import torch
import torch.nn as nn
# from torchmetrics.functional import image_gradients
from torchvision.transforms import transforms
import torch.nn.functional as F
from light_training.loss.my_utils import *
from math import exp
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
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
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return torch.abs(ssim_map.mean())
    else:
        return torch.abs(ssim_map.mean(1).mean(1).mean(1))
    


class SSIM(torch.nn.Module):
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
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def Contrast(img1, img2, window_size=11, channel=1):
    window = create_window(window_size, channel)    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq

    return sigma1_sq, sigma2_sq

    
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
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


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

class grad_loss(nn.Module):
    '''
    image gradient loss
    '''
    def __init__(self, device, vis = False, type = "sobel"):

        super(grad_loss, self).__init__()
        
        # only use sobel filter now
        if type == "sobel":
            kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
            kernel_y = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        # do not want update these weights
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).to(device)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).to(device)
        
        self.vis = vis
    
    def forward(self, x, y):
        # conv2d to find image gradient in x direction and y direction
        # of input image x and image y
        grad_xx = F.conv2d(x, self.weight_x)
        grad_xy = F.conv2d(x, self.weight_y)
        grad_yx = F.conv2d(y, self.weight_x)
        grad_yy = F.conv2d(y, self.weight_y)

        if self.vis:
            return grad_xx, grad_xy, grad_yx, grad_yy
        
        # total image gradient, in dx and dy direction for image X and Y
        # gradientX = torch.abs(grad_xx) + torch.abs(grad_xy)
        # gradientY = torch.abs(grad_yx) + torch.abs(grad_yy)
        x_diff = ((torch.abs(grad_xx) - torch.abs(grad_yx)) ** 2).mean()
        y_diff = ((torch.abs(grad_xy) - torch.abs(grad_yy)) ** 2).mean()
        
        # mean squared frobenius norm (||.||_F^2)
        #grad_f_loss = torch.mean(torch.pow(torch.norm((gradientX - gradientY), p = "fro"), 2))
        grad_f_loss = x_diff + y_diff
        return grad_f_loss


def l1_loss(predicted, target):
    """
    To compute L1 loss using predicted and target
    """
    return torch.abs(predicted - target).mean()


def mse_loss(predicted, target):
    """
    To compute L2 loss between predicted and target
    """
    return torch.pow((predicted - target), 2).mean()
    #return torch.mean(torch.pow(torch.norm((predicted - target), p="fro"), 2))


def img_gradient(img: torch.Tensor):
    """
    Input: one PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
    Output: image gradient (2 x C x H x W)
    """
    # trans = transforms.ToTensor()
    # # a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    # img_tensor = trans(img)
    # # reshape to [N, C, H, W]
    # img_tensor = img_tensor.reshape((1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2]))
    dy, dx = image_gradients(img)
    dy, dx = dy.squeeze(), dx.squeeze()
    dxy = torch.stack((dx, dy), axis=0)
    return dxy


def gradient_loss(predicted, target):
    """
    compute image gradient loss between predicted and target
    """
    # grad_p = np.gradient(predicted)
    # grad_t = np.gradient(target)
    grad_p = img_gradient(predicted)
    grad_t = img_gradient(target)
    return torch.pow((grad_p - grad_t), 2).mean()


def perceptual_loss(vgg, predicted, target, block_idx, device):
    """
    compute perceptual loss between predicted and target
    """
    p_loss = Percep_loss(vgg, block_idx, device)
    return p_loss(predicted, target)


def ssim_loss(predicted, target):
    """
    ssim loss
    """
    ssim_metric = SSIM().cuda()
    return 1 - ssim_metric(predicted, target)

def loss_func(predicted, target, lambda1, lambda2, block_idx, device):
    """
    Implement the loss function in our proposal
    Loss = a variant of the MSE loss + perceptual loss
    """
    loss = mse_loss(predicted, target) + lambda1 * gradient_loss(predicted, target)
    +lambda2 * perceptual_loss(predicted, target, block_idx, device)
    return loss


def pair_loss(predicted, target1, target2, device):
    """
    pair loss
    """
    img_grad_loss = grad_loss(device)
    ssim = SSIM(size_average = False).to(device)
    loss1 = l1_loss(ssim(predicted, target1), ssim(predicted, target2))
    #loss2 = l1_loss((1 - img_grad_loss(predicted, target1)), (1 - img_grad_loss(predicted, target2)))
    return loss1#, loss2

def loss_func2(predicted, target1, target2, device):
    """
    same as loss_func, except the gradient loss is change to grad_loss() class
    """
    img_grad_loss = grad_loss(device)
    #L1_charbonnier = L1_Charbonnier_loss()
    #reg_loss = L1_charbonnier(predicted, target)
    reg_loss1 = l1_loss(predicted, target1)
    reg_loss2 = l1_loss(predicted, target2)
    # img_grad_dif1 = img_grad_loss(predicted, target1)
    # img_grad_dif2 = img_grad_loss(predicted, target2)
    ssim_loss1 = ssim_loss(predicted, target1)
    ssim_loss2 = ssim_loss(predicted, target2)
    #percep = perceptual_loss(vgg, predicted, target, block_idx, device)
    pair1 = pair_loss(predicted, target1, target2, device)
    loss = reg_loss1 + reg_loss2 + ssim_loss1 + ssim_loss2 + pair1# + pair2
    return loss, pair1#, pair2


def loss_function_l2(predicted, target):
    loss = nn.MSELoss()
    return loss(predicted, target)

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        Loss_SSIM = 0.5 * ssim(image_A, image_fused) + 0.5 * ssim(image_B, image_fused)
        return Loss_SSIM

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_fused_Y = image_fused[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        # gradient_A = TF.gaussian_blur(gradient_A, 3, [1, 1])
        gradient_B = self.sobelconv(image_B_Y)
        # gradient_B = TF.gaussian_blur(gradient_B, 3, [1, 1])
        gradient_fused = self.sobelconv(image_fused_Y)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return torch.abs(Loss_intensity)

class img_grad(nn.Module):
    def __init__(self, device, b_size) -> None:
        super(img_grad, self).__init__()
        
        self.device = device
        self.x_gradient_filter = torch.Tensor(
            [
                #[[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                # [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                # [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            ]
        ).to(self.device)
        self.x_gradient_filter = self.x_gradient_filter.unsqueeze(0)#.repeat(b_size,1,1,1)

        self.y_gradient_filter = torch.Tensor(
            [
                #[[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                # [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                # [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                [[1., 2., 1.],[0, 0, 0],[-1., -2., -1.]],
            ]
        ).to(self.device)
        self.y_gradient_filter = self.y_gradient_filter.unsqueeze(0)#.repeat(b_size,1,1,1)

        self.weight_x = nn.Parameter(data=self.x_gradient_filter, requires_grad=False).to(device)
        self.weight_y = nn.Parameter(data=self.y_gradient_filter, requires_grad=False).to(device)
    
    def forward(self, image1, image2, image_fused):
        '''
        input should be 2d images with shape [batch, 1, h, w]
        batch should be 3
        '''
        grad_x1_x = F.conv2d(image1, self.weight_x, padding=1)
        grad_x2_x = F.conv2d(image2, self.weight_x, padding=1)
        
        grad_x1_y = F.conv2d(image1, self.weight_y, padding=1)
        grad_x2_y = F.conv2d(image2, self.weight_y, padding=1)

        grad_f_x = F.conv2d(image_fused, self.weight_x, padding=1)
        grad_f_y = F.conv2d(image_fused, self.weight_y, padding=1)

        x_diff = ((torch.abs(grad_f_x - torch.max(grad_x1_x, grad_x2_x)) ** 2)).mean()
        y_diff = ((torch.abs(grad_f_y - torch.max(grad_x1_y, grad_x2_y)) ** 2)).mean()

        return x_diff + y_diff



def img_grad_loss_3d(img1, img2, img_fuse, device):
    '''
    compute img grad loss in 3d
    true_img and rec_img: should have size [batch, 1, 128, 128, 128], channel first
    '''

    _batch_size = img1.shape[0]
    img_grad_loss = img_grad(device, _batch_size)

    total = 0

    for i in range(img1.shape[2]):
        temp = 0
        # now get 3d loss
        img_aX = img1[:,:,i,:,:]
        img_bX = img2[:,:,i,:,:]
        img_fuseX = img_fuse[:,:,i,:,:]
        img_grad_lossX = img_grad_loss(img_aX, img_bX, img_fuseX)
        temp += img_grad_lossX

        img_aY = img1[:,:,:,i,:]
        img_bY = img2[:,:,:,i,:]
        img_fuseX = img_fuse[:,:,:,i,:]
        img_grad_lossY = img_grad_loss(img_aY, img_bY, img_fuseX)
        temp += img_grad_lossY

        img_aZ = img1[:,:,:,:,i]
        img_bZ = img2[:,:,:,:,i]
        img_fuseZ = img_fuse[:,:,:,:,i]
        img_grad_lossZ = img_grad_loss(img_aZ, img_bZ, img_fuseZ)
        temp += img_grad_lossZ

        #temp /= 3
        total += temp
    
    return total / img1.shape[2]

class fusion_loss_med(nn.Module):
    def __init__(self):
        super(fusion_loss_med, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

        # print(1)
    def forward(self, image_A, image_B, image_fused):
        # image_A represents MRI image
        loss_l1 = 2 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 10 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 5 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM


class fusion_loss_med3D(nn.Module):
    def __init__(self):
        super(fusion_loss_med3D, self).__init__()
        # self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = SSIM3D()

        # print(1)
    def forward(self, image_A, image_B, image_fused):
        # image_A represents MRI image
        loss_l1 = 2 * torch.abs(self.L_Inten(image_A, image_B, image_fused))
        loss_gradient = 10 * img_grad_loss_3d(image_A, image_B, image_fused, 'cuda')
        loss_ssim_imga = torch.abs(self.L_SSIM(image_A, image_fused))
        # print("loss_ssim_imga", loss_ssim_imga)
        loss_ssim_imgb = torch.abs(self.L_SSIM(image_B, image_fused))
        # print("loss_ssim_imgb", loss_ssim_imgb)
        loss_SSIM = 1 * torch.abs(0.5*(1-loss_ssim_imga) + 0.5*(1-loss_ssim_imgb))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM

class fusion_loss_medv2(nn.Module):
    def __init__(self):
        super(fusion_loss_medv2, self).__init__()
        # self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = SSIM3D()

        # print(1)
    def forward(self, image_A, image_fused):
        # image_A represents MRI image
        loss_l1 = 2 * torch.abs(self.L_Inten(image_A, image_B, image_fused))
        loss_gradient = 10 * img_grad_loss_3d(image_A, image_B, image_fused, 'cuda')
        loss_ssim_imga = torch.abs(self.L_SSIM(image_A, image_fused))
        # print("loss_ssim_imga", loss_ssim_imga)
        loss_ssim_imgb = torch.abs(self.L_SSIM(image_B, image_fused))
        # print("loss_ssim_imgb", loss_ssim_imgb)
        loss_SSIM = 1 * torch.abs(0.5*(1-loss_ssim_imga) + 0.5*(1-loss_ssim_imgb))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM