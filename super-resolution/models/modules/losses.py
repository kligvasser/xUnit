import torch
import torch.nn as nn
import torch.nn.functional as F
from .misc import shave_edge
from ..vgg import MultiVGGFeaturesExtractor

class RangeLoss(nn.Module):
    def __init__(self, min_value=0., max_value=1., invalidity_margins=None):
        super(RangeLoss, self).__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.invalidity_margins = invalidity_margins

    def forward(self, inputs):
        if self.invalidity_margins:
            inputs = shave_edge(inputs, self.invalidity_margins, self.invalidity_margins)
        loss = (F.relu(self.min_value - inputs) + F.relu(inputs - self.max_value)).mean()
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self, features_to_compute, criterion=torch.nn.L1Loss(), shave_edge=None):
        super(PerceptualLoss, self).__init__()
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute, requires_grad=False, shave_edge=shave_edge).eval()

    def forward(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        loss = 0
        for key in inputs_fea.keys():
            loss += self.criterion(inputs_fea[key], targets_fea[key].detach())

        return loss

class TexturalLoss(nn.Module):
    def __init__(self, features_to_compute, criterion=torch.nn.L1Loss(), shave_edge=None):
        super(TexturalLoss, self).__init__()
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute, requires_grad=False, use_input_norm=True, shave_edge=shave_edge).eval()

    def forward(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        loss = 0
        for key in inputs_fea.keys():
            inputs_gram = self._gram_matrix(inputs_fea[key])
            with torch.no_grad():
                targets_gram = self._gram_matrix(targets_fea[key]).detach()

            loss += self.criterion(inputs_gram, targets_gram)

        return loss

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)