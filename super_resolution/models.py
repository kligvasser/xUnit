### imports ###
import torch.nn as nn
import torch

### models ###
class abselute(nn.Module):
    def forward(self, input):
        return torch.abs(input)

class distances(nn.Module):
    def forward(self,input):
        mul = torch.mul(input,input)
        return torch.exp(-mul)

class multipication(nn.Module):
    def forward(self, inputs):
        return torch.mul(inputs[0],inputs[1])

class modulecell(nn.Module):
    def __init__(self,in_channels=1,out_channels=64,kernel_size=3,skernel_size=9):
        super(modulecell,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=((kernel_size-1)//2),bias=True))
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

class xSRCNNf(nn.Module):
    def __init__(self):
        super(xSRCNNf,self).__init__()
        self.md1 = nn.Sequential(
            modulecell(in_channels=1,out_channels=64,kernel_size=9))
        self.md2 = nn.Sequential(
            modulecell(in_channels=64,out_channels=32,kernel_size=3))
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
            modulecell(in_channels=1,out_channels=42,kernel_size=9))
        self.md2 = nn.Sequential(
            modulecell(in_channels=42,out_channels=32,kernel_size=5))
        self.joints = nn.Conv2d(32,1,kernel_size=5,padding=2)

    def forward(self,x):
        x = self.md1(x)
        x = self.md2(x)
        x = self.joints(x)
        return x

### main ###

