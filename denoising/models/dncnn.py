import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

__all__ = ['g_dncnn', 'd_dncnn']

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

class GenBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, bias=True):
        super(GenBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size // 2), bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        initialize_weights([self.conv, self.bn], 0.02)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
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
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks):
        super(Generator, self).__init__()

        # image to features
        self.image_to_features = GenBlock(in_channels=in_channels, out_channels=num_features)

        # features
        blocks = []
        for _ in range(gen_blocks):
            blocks.append(GenBlock(in_channels=num_features, out_channels=num_features, bias=False))
        self.features = nn.Sequential(*blocks)

        # features to image
        self.features_to_image = nn.Conv2d(in_channels=num_features, out_channels=in_channels, kernel_size=5, padding=2)
        initialize_weights([self.features_to_image], 0.02)

    def forward(self, x):
        r = x
        x = self.image_to_features(x)
        x = self.features(x)
        x = self.features_to_image(x)
        x += r
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks):
        super(Discriminator, self).__init__()

        # image to features
        self.image_to_features = DisBlock(in_channels=in_channels, out_channels=num_features, bias=True, normalization=False)

        # features
        blocks = []
        for i in range(0, dis_blocks - 1):
            blocks.append(DisBlock(in_channels=num_features * min(pow(2, i), 8), out_channels=num_features * min(pow(2, i + 1), 8), bias=False, normalization=False))
        self.features = nn.Sequential(*blocks)

        # classifier
        self.classifier = nn.Conv2d(in_channels=num_features * min(pow(2, dis_blocks - 1), 8), out_channels=1, kernel_size=4, padding=0)

    def forward(self, x):
        x = self.image_to_features(x)
        x = self.features(x)
        x = self.classifier(x)
        x = x.flatten(start_dim=1).mean(dim=-1)
        return x

class SNDiscriminator(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks):
        super(SNDiscriminator, self).__init__()

        # image to features
        self.image_to_features = DisBlock(in_channels=in_channels, out_channels=num_features, bias=True, normalization=True)

        # features
        blocks = []
        for i in range(0, dis_blocks - 1):
            blocks.append(DisBlock(in_channels=num_features * min(pow(2, i), 8), out_channels=num_features * min(pow(2, i + 1), 8), bias=False, normalization=True))
        self.features = nn.Sequential(*blocks)

        # classifier
        self.classifier = SpectralNorm(nn.Conv2d(in_channels=num_features * min(pow(2, dis_blocks - 1), 8), out_channels=1, kernel_size=4, padding=0))

    def forward(self, x):
        x = self.image_to_features(x)
        x = self.features(x)
        x = self.classifier(x)
        x = x.flatten(start_dim=1).mean(dim=-1)
        return x

def g_dncnn(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('gen_blocks', 8)
    config.setdefault('dis_blocks', 5)

    return Generator(**config)

def d_dncnn(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('gen_blocks', 8)
    config.setdefault('dis_blocks', 5)

    return Discriminator(**config)