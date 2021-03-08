import torch
import torchvision
import torch.nn as nn
from models.modules.misc import shave_edge
from collections import OrderedDict

names = {'vgg19': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                   'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                   'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
                   'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                   'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
                   'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                   'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
                   'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'],

         'vgg19_bn': ['conv1_1', 'bn1_1', 'relu1_1', 'conv1_2', 'bn1_2', 'relu1_2', 'pool1',
                      'conv2_1', 'bn2_1', 'relu2_1', 'conv2_2', 'bn2_2', 'relu2_2', 'pool2',
                      'conv3_1', 'bn3_1', 'relu3_1', 'conv3_2', 'bn3_2', 'relu3_2',
                      'conv3_3', 'bn3_3', 'relu3_3', 'conv3_4', 'bn3_4', 'relu3_4', 'pool3',
                      'conv4_1', 'bn4_1', 'relu4_1', 'conv4_2', 'bn4_2', 'relu4_2',
                      'conv4_3', 'bn4_3', 'relu4_3', 'conv4_4', 'bn4_4', 'relu4_4', 'pool4',
                      'conv5_1', 'bn5_1', 'relu5_1', 'conv5_2', 'bn5_2', 'relu5_2',
                      'conv5_3', 'bn5_3', 'relu5_3', 'conv5_4', 'bn5_4', 'relu5_4', 'pool5']
         }


class VGGFeaturesExtractor(nn.Module):
    def __init__(self, feature_layer='conv5_4', use_bn=False, use_input_norm=True, requires_grad=False):
        super(VGGFeaturesExtractor, self).__init__()
        self.use_input_norm = use_input_norm

        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        layer_index = names['vgg19_bn'].index(feature_layer) if use_bn else names['vgg19'].index(feature_layer)
        self.features = nn.Sequential(*list(model.features.children())[:(layer_index + 1)])

        if not requires_grad:
            for k, v in self.features.named_parameters():
                v.requires_grad = False
            self.features.eval()

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

class MultiVGGFeaturesExtractor(nn.Module):
    def __init__(self, target_features=('relu1_1', 'relu2_1', 'relu3_1'), use_bn=False, use_input_norm=True, requires_grad=False, shave_edge=None):
        super(MultiVGGFeaturesExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        self.target_features = target_features
        self.shave_edge = shave_edge

        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
            names_key = 'vgg19_bn'
        else:
            model = torchvision.models.vgg19(pretrained=True)
            names_key = 'vgg19'

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        self.target_indexes = [names[names_key].index(k) for k in self.target_features]
        self.features = nn.Sequential(*list(model.features.children())[:(max(self.target_indexes) + 1)])

        if not requires_grad:
            for k, v in self.features.named_parameters():
                v.requires_grad = False
            self.features.eval()

    def forward(self, x):
        if self.shave_edge:
            x = shave_edge(x, self.shave_edge, self.shave_edge)

        # assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = OrderedDict()
        for key, layer in self.features._modules.items():
            x = layer(x)
            if int(key) in self.target_indexes:
                output.update({self.target_features[self.target_indexes.index(int(key))]: x})
        return output