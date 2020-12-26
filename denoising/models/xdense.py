import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
from .modules.activations import xUnitD
from .modules.misc import center_crop

__all__ = ['g_xdense', 'd_xdense']

def initialize_weights(net, scale=1.):
    if not isinstance(net, list):
        net = [net]
    for layer in net:
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)

class xModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(xModule, self).__init__()
        # features
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            xUnitD(num_features=out_channels, batch_norm=True),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        x = self.features(x)
        return x

class xDenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(xDenseLayer, self).__init__()
        # features
        self.features = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,out_channels=bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=bn_size * growth_rate),
            nn.ReLU(inplace=True),
            xModule(in_channels=bn_size * growth_rate, out_channels=growth_rate)
        )

    def forward(self, x):
        f = self.features(x)
        return torch.cat([x, f], dim=1)

class xDenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, bn_size):
        super(xDenseBlock, self).__init__()
        # features
        blocks = []
        for i in range(num_layers):
            blocks.append(xDenseLayer(in_channels=in_channels + growth_rate*i, growth_rate=growth_rate, bn_size=bn_size))
        self.features = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.features(x)
        return x

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        # features
        self.features = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class DisBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True, normalization=False):
        super(DisBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)

        initialize_weights([self.conv1, self.conv2], 0.1)

        if normalization:
            self.conv1 = SpectralNorm(self.conv1)
            self.conv2 = SpectralNorm(self.conv2)

    def forward(self, x):
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        return x

class Generator(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks, growth_rate, bn_size):
        super(Generator, self).__init__()

        # image to features
        self.image_to_features = xModule(in_channels=in_channels, out_channels=num_features)

        # features
        blocks = []
        self.num_features = num_features
        for i, num_layers in enumerate(gen_blocks):
            blocks.append(xDenseBlock(in_channels=self.num_features, num_layers=num_layers, growth_rate=growth_rate, bn_size=bn_size))
            self.num_features += num_layers * growth_rate
            
            if i != len(gen_blocks) - 1:
                blocks.append(Transition(in_channels=self.num_features, out_channels=self.num_features // 2))
                self.num_features = self.num_features // 2

        self.features = nn.Sequential(*blocks)

        # features to image
        self.features_to_image = nn.Conv2d(in_channels=self.num_features, out_channels=in_channels, kernel_size=5, padding=2)

    def forward(self, x):
        r = x
        x = self.image_to_features(x)
        x = self.features(x)
        x = self.features_to_image(x)
        x += r
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks, growth_rate, bn_size):
        super(Discriminator, self).__init__()
        self.crop_size = 4 * pow(2, dis_blocks)

        # image to features
        self.image_to_features = DisBlock(in_channels=in_channels, out_channels=num_features, bias=True, normalization=False)

        # features
        blocks = []
        for i in range(0, dis_blocks - 1):
            blocks.append(DisBlock(in_channels=num_features * min(pow(2, i), 8), out_channels=num_features * min(pow(2, i + 1), 8), bias=False, normalization=False))
        self.features = nn.Sequential(*blocks)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features * min(pow(2, dis_blocks - 1), 8) * 4 * 4, 100),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = center_crop(x, self.crop_size, self.crop_size)
        x = self.image_to_features(x)
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

class SNDiscriminator(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks, growth_rate, bn_size):
        super(SNDiscriminator, self).__init__()
        self.crop_size = 4 * pow(2, dis_blocks)

        # image to features
        self.image_to_features = DisBlock(in_channels=in_channels, out_channels=num_features, bias=True, normalization=True)

        # features
        blocks = []
        for i in range(0, dis_blocks - 1):
            blocks.append(DisBlock(in_channels=num_features * min(pow(2, i), 8), out_channels=num_features * min(pow(2, i + 1), 8), bias=False, normalization=True))
        self.features = nn.Sequential(*blocks)

        # classifier
        self.classifier = nn.Sequential(
            SpectralNorm(nn.Linear(num_features * min(pow(2, dis_blocks - 1), 8) * 4 * 4, 100)),
            nn.LeakyReLU(negative_slope=0.1),
            SpectralNorm(nn.Linear(100, 1))
        )

    def forward(self, x):
        x = center_crop(x, self.crop_size, self.crop_size)
        x = self.image_to_features(x)
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

def g_xdense(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('gen_blocks', [4, 6, 8])
    config.setdefault('dis_blocks', 4)
    config.setdefault('growth_rate', 16)
    config.setdefault('bn_size', 2)

    _ = config.pop('spectral', False)

    return Generator(**config)

def d_xdense(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('gen_blocks', [4, 6, 8])
    config.setdefault('dis_blocks', 4)
    config.setdefault('growth_rate', 16)
    config.setdefault('bn_size', 2)

    sn = config.pop('spectral', False)

    if sn:
        return SNDiscriminator(**config)
    else:
        return Discriminator(**config)
