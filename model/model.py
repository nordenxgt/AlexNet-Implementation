import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernal_size: int, 
        stride: int = 1, 
        norm: bool = False, 
        pool: bool = False
    ) -> None:
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernal_size, stride=stride))
        if norm: layers.append(nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2))
        if pool: layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        layers.append(nn.ReLU())
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)

class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: bool = True) -> None:
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, out_features))
        if dropout: layers.append(nn.Dropout(p=0.5))
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class AlexNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 96, kernal_size=11, stride=4, norm=True, pool=True)
        self.conv2 = ConvBlock(96, 256, kernal_size=5, norm=True, pool=True)
        self.conv3 = ConvBlock(256, 384, kernal_size=3)
        self.conv4 = ConvBlock(384, 256, kernal_size=3)
        self.conv5 = ConvBlock(256, 256, kernal_size=3, pool=True)
        self.fc1 = LinearBlock(256*6*6, 4096)
        self.fc2 = LinearBlock(4096, 4096)
        self.fc3 = LinearBlock(4096, num_classes, dropout=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x