import torch.nn as nn

class xUnit(nn.Module):
    def __init__(self, num_features=64, kernel_size=7, batch_norm=False):
        super(xUnit, self).__init__()
        # xUnit
        self.features = nn.Sequential(
            nn.BatchNorm2d(num_features=num_features) if batch_norm else Identity(),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=(kernel_size // 2), groups=num_features),
            nn.BatchNorm2d(num_features=num_features) if batch_norm else Identity(),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.features(x)
        r = x * a
        return r

class xUnitS(nn.Module):
    def __init__(self, num_features=64, kernel_size=7, batch_norm=False):
        super(xUnitS, self).__init__()
        # slim xUnit
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=(kernel_size // 2), groups=num_features),
            nn.BatchNorm2d(num_features=num_features) if batch_norm else Identity(),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.features(x)
        r = x * a
        return r

class xUnitD(nn.Module):
    def __init__(self, num_features=64, kernel_size=7, batch_norm=False):
        super(xUnitD, self).__init__()
        # dense xUnit
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=num_features) if batch_norm else Identity(),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=(kernel_size // 2), groups=num_features),
            nn.BatchNorm2d(num_features=num_features) if batch_norm else Identity(),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.features(x)
        r = x * a
        return r

class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
