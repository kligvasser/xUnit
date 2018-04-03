### imports ###
import torch.nn as nn
import torch

### models ###
class Gaussian(nn.Module):
    def forward(self,input):
        return torch.exp(-torch.mul(input,input))

class Modulecell(nn.Module):
    def __init__(self,in_channels=1,out_channels=64,kernel_size=3,skernel_size=9):
        super(Modulecell,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=((kernel_size-1)//2),bias=True))
        self.module = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=skernel_size,stride=1,padding=((skernel_size-1)//2),groups=out_channels),
            nn.BatchNorm2d(out_channels),
            Gaussian())
    def forward(self,x):
        x1 = self.features(x)
        x2 = self.module(x1)
        x = torch.mul(x1,x2)
        return x

class xResidualBlock(nn.Module):
    def __init__(self,in_channels=64,k=3,n=64,s=1):
        super(xResidualBlock,self).__init__()
        self.md = Modulecell(in_channels,n,k)
        self.conv2 = nn.Conv2d(n,n,k,stride=s,padding=1)
        self.bn1 = nn.BatchNorm2d(n)
    def forward(self,x):
        y = self.md(x)
        return self.bn1(self.conv2(y))+x

class UpsampleBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpsampleBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()
    def forward(self,x):
        return self.prelu(self.shuffler(self.conv(x)))

class xSRCNNf(nn.Module):
    def __init__(self):
        super(xSRCNNf,self).__init__()
        self.md1 = nn.Sequential(
            Modulecell(in_channels=1,out_channels=64,kernel_size=9))
        self.md2 = nn.Sequential(
            Modulecell(in_channels=64,out_channels=32,kernel_size=3))
        self.joints = nn.Conv2d(32,1,kernel_size=5,padding=2)
    def forward(self,x):
        x = self.md1(x)
        x = self.md2(x)
        x = self.joints(x)
        return x

class xSRCNNc(nn.Module):
    def __init__(self):
        super(xSRCNNc,self).__init__()
        self.md1 = nn.Sequential(
            Modulecell(in_channels=1,out_channels=42,kernel_size=9))
        self.md2 = nn.Sequential(
            Modulecell(in_channels=42,out_channels=32,kernel_size=5))
        self.joints = nn.Conv2d(32,1,kernel_size=5,padding=2)
    def forward(self,x):
        x = self.md1(x)
        x = self.md2(x)
        x = self.joints(x)
        return x

class xSRResNet(nn.Module):
    def __init__(self,n_residual_blocks=10,upsample_factor=4):
        super(xSRResNet,self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor
        self.conv1 = nn.Conv2d(3,64,9,stride=1,padding=4)
        self.prelu1 = nn.PReLU()
        for i in range(self.n_residual_blocks):
            self.add_module('xresidual_block'+str(i+1),xResidualBlock())
        self.conv2 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        for i in range(upsample_factor//2):
            self.add_module('upsample'+str(i+1),UpsampleBlock(64,256))
        self.conv3 = nn.Conv2d(64,3,9,stride=1,padding=4)
    def forward(self,x):
        x = self.prelu1(self.conv1(x))
        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('xresidual_block'+str(i+1))(y)
        x = self.bn2(self.conv2(y))+x
        for i in range(self.upsample_factor//2):
            x = self.__getattr__('upsample'+str(i+1))(x)
        return self.conv3(x)

### main ###

