### imports ###
import torch.nn as nn
import torch

### models ###
class distances(nn.Module):
    def forward(self,input):
        return torch.exp(-torch.mul(input,input))

class multipication(nn.Module):
    def forward(self,inputs):
        return torch.mul(inputs[0],inputs[1])

class modulecell(nn.Module):
    def __init__(self,in_channels=1,out_channels=64,kernel_size=3,skernel_size=9):
        super(modulecell,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=((kernel_size-1)//2)))
        self.module = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=skernel_size,stride=1,padding=((skernel_size-1)//2),groups=out_channels),
            nn.BatchNorm2d(out_channels),
            distances())
        self.multi = nn.Sequential(
            multipication())
    def forward(self,x):
        x1 = self.features(x)
        x2 = self.module(x1)
        x = self.multi([x1,x2])
        return x

class xDnCNN(nn.Module):
    def __init__(self,channels=64):
        super(xDnCNN,self).__init__()
        self.md1 = nn.Sequential(
            modulecell(in_channels=1,out_channels=channels,kernel_size=3))
        self.md2 = nn.Sequential(
            nn.BatchNorm2d(channels),
            modulecell(in_channels=channels,out_channels=channels,kernel_size=3))
        self.md3 = nn.Sequential(
            nn.BatchNorm2d(channels),
            modulecell(in_channels=channels,out_channels=channels,kernel_size=3))
        self.md4 = nn.Sequential(
            nn.BatchNorm2d(channels),
            modulecell(in_channels=channels,out_channels=channels,kernel_size=3))
        self.md5 = nn.Sequential(
            nn.BatchNorm2d(channels),
            modulecell(in_channels=channels,out_channels=channels,kernel_size=3))
        self.md6 = nn.Sequential(
            nn.BatchNorm2d(channels),
            modulecell(in_channels=channels,out_channels=channels,kernel_size=3))
        self.md7 = nn.Sequential(
            nn.BatchNorm2d(channels),
            modulecell(in_channels=channels,out_channels=channels,kernel_size=3))
        self.md8 = nn.Sequential(
            nn.BatchNorm2d(channels),
            modulecell(in_channels=channels,out_channels=channels,kernel_size=3))
        self.joints = nn.Conv2d(channels,1,kernel_size=3,padding=1)
    def forward(self,x):
        x = self.md1(x)
        x = self.md2(x)
        x = self.md3(x)
        x = self.md4(x)
        x = self.md5(x)
        x = self.md6(x)
        x = self.md7(x)
        x = self.md8(x)
        x = self.joints(x)
        return x

### main ###
