import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleX2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UpsampleX2, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=(out_channels * 4), kernel_size=kernel_size, padding=(kernel_size // 2))
        self.shuffler = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.lrelu(self.shuffler(self.conv(x)))

def center_crop(x, height, width):
    crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
    crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)

    return F.pad(x, [
        crop_w.ceil().int()[0], crop_w.floor().int()[0],
        crop_h.ceil().int()[0], crop_h.floor().int()[0],
    ])

def shave_edge(x, shave_h, shave_w):
    return F.pad(x, [-shave_w, -shave_w, -shave_h, -shave_h])

def shave_modulo(x, factor):
    shave_w = x.size(-1) % factor
    shave_h = x.size(-2) % factor
    return F.pad(x, [0, -shave_w, 0, -shave_h])

if __name__ == "__main__":
    x = torch.randn(1, 2, 4, 6)
    y = shave_edge(x, 1, 2)
    print(x)
    print(y)