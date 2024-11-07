import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips
import matplotlib.pyplot as plt

def imshow_tensor(tensor, title=None, cmap=None):
    """
    Display a tensor using matplotlib.
    
    Args:
        tensor (torch.Tensor): The image tensor (shape: [C, H, W] or [H, W, C]).
        title (str): Title for the plot.
        cmap (str): Colormap for single-channel images (e.g., "gray").
    """
    # Move to CPU if necessary
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Squeeze batch dimension if present
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    # Handle channel-first format
    if tensor.dim() == 3 and tensor.size(0) in [1, 3]:  # [C, H, W]
        tensor = tensor.permute(1, 2, 0)  # [H, W, C]

    # Ensure the tensor is in the range [0, 1]
    tensor = tensor.detach().clamp(0, 1)

    # Plot the tensor
    plt.imshow(tensor.numpy(), cmap=cmap)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def computeMask(img1):
    magenta = torch.tensor([1, 0, 1], dtype=torch.float32).view(1, 3, 1, 1).cuda()
    tolerance = 0.1
    magenta_mask = torch.all(torch.abs(img1 - magenta) < tolerance, dim=1, keepdim=True)
    return ~magenta_mask[0]

def psnr(img1, img2, mask):
    mask = mask.expand_as(img1)
    mse = ((((img1 - img2)) ** 2) * mask).view(img1.shape[0], -1)[mask.view(mask.shape[0], -1)].mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    img1 = img1 * mask
    img2 = img2 * mask

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

loss_fn_vgg = lpips.LPIPS(net='vgg').to("cuda")
loss_fn_vgg.eval()
def lpips_(img1, img2, mask):
    img1 = img1 * mask
    img2 = img2 * mask
    return torch.mean(loss_fn_vgg(img1, img2))