import torch.nn as nn
from math import log
from torch.nn import functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
from .modules.activations import xUnitS
from .modules.misc import UpsampleX2, center_crop

__all__ = ['g_xsrgan', 'd_xsrgan', 'd_xsrgan_ad']

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
    def __init__(self, num_features=64, kernel_size=3, bias=False):
        super(GenBlock, self).__init__()
        self.xunit = xUnitS(num_features=num_features)
        self.conv1 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=(kernel_size // 2), bias=bias)
        self.conv2 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=(kernel_size // 2), bias=bias)

        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        residual = x
        x = self.conv2(self.xunit(self.conv1(x)))
        x = residual + x
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
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks, scale):
        super(Generator, self).__init__()
        self.scale = scale
        # image to features
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.image_to_features = nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=5, padding=2)

        # features
        blocks = []
        for _ in range(gen_blocks):
            blocks.append(GenBlock(num_features=num_features))
        self.features = nn.Sequential(*blocks)

        # upsampling
        blocks = []
        for _ in range(int(log(scale, 2))):
            block = UpsampleX2(in_channels=num_features, out_channels=num_features)
            blocks.append(block)
        self.usample = nn.Sequential(*blocks)

        # features to image
        self.hrconv = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)
        self.features_to_image = nn.Conv2d(in_channels=num_features, out_channels=in_channels, kernel_size=5, padding=2)

        # init weights
        initialize_weights([self.image_to_features, self.hrconv, self.features_to_image], 0.1)

    def forward(self, x):
        r = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.lrelu(self.image_to_features(x))
        x = self.features(x)
        x = self.usample(x)
        x = self.features_to_image(self.lrelu(self.hrconv(x)))
        x = x + r
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks, scale):
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

class DiscriminatorAdaptive(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks, scale):
        super(DiscriminatorAdaptive, self).__init__()
        self.crop_size = 4 * pow(2, dis_blocks)

        # image to features
        self.image_to_features = DisBlock(in_channels=in_channels, out_channels=num_features, bias=True, normalization=False)

        # features
        blocks = []
        for i in range(0, dis_blocks - 1):
            blocks.append(DisBlock(in_channels=num_features * min(pow(2, i), 8), out_channels=num_features * min(pow(2, i + 1), 8), bias=False, normalization=False))
        self.features = nn.Sequential(*blocks)

        # pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features * min(pow(2, dis_blocks - 1), 8), num_features),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(num_features, 1)
        )

    def forward(self, x):
        x = center_crop(x, self.crop_size, self.crop_size)
        x = self.image_to_features(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

class SNDiscriminator(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks, scale):
        super(SNDiscriminator, self).__init__()
        self.crop_size = 4 * pow(2, dis_blocks)

        # image to features
        self.image_to_features = DisBlock(in_channels=in_channels, out_channels=num_features, bias=True, normalization=True)

        # features
        blocks = []
        for i in range(0, dis_blocks - 1):
            blocks.append(DisBlock(in_channels=num_features * min(pow(2, i), 8), out_channels=num_features * min(pow(2, i + 1), 8), bias=False, normalization=True))
        self.features = nn.Sequential(*blocks)

        # pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

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

class SNDiscriminatorAdaptive(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks, scale):
        super(SNDiscriminatorAdaptive, self).__init__()
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
            SpectralNorm(nn.Linear(num_features * min(pow(2, dis_blocks - 1), 8), num_features)),
            nn.LeakyReLU(negative_slope=0.1),
            SpectralNorm(nn.Linear(num_features, 1))
        )

    def forward(self, x):
        x = center_crop(x, self.crop_size, self.crop_size)
        x = self.image_to_features(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

def g_xsrgan(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('gen_blocks', 10)
    config.setdefault('dis_blocks', 5)
    config.setdefault('scale', 4)

    _ = config.pop('spectral', False)

    return Generator(**config)

def d_xsrgan(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('gen_blocks', 10)
    config.setdefault('dis_blocks', 5)
    config.setdefault('scale', 4)

    sn = config.pop('spectral', False)

    if sn:
        return SNDiscriminator(**config)
    else:
        return Discriminator(**config)

def d_xsrgan_ad(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('gen_blocks', 10)
    config.setdefault('dis_blocks', 5)
    config.setdefault('scale', 4)

    sn = config.pop('spectral', False)

    if sn:
        return SNDiscriminatorAdaptive(**config)
    else:
        return DiscriminatorAdaptive(**config)
