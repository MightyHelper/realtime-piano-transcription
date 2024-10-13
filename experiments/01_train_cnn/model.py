from enum import StrEnum

import torch
import torch.nn.functional as F

class ResNetBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride, kernel_size // 2),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2)

    def forward(self, x):
        relu = F.relu(self.block(x) + self.skip(x))
        return relu

class ResNetBlockTranspose(torch.nn.Module):

        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
            super().__init__()

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, kernel_size // 2),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()
            )
            self.skip = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2)

        def forward(self, x):
            relu = F.relu(self.block(x) + self.skip(x))
            return relu

def get_model(n_predict: int, device: torch.device | str):
    model = torch.nn.Sequential(
        torch.nn.LazyConv2d(16, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.LazyConv2d(32, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.LazyConv2d(64, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.LazyConv2d(128, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(),
        torch.nn.LazyConv2d(512, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(),
        torch.nn.LazyConvTranspose2d(512, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(),
        torch.nn.LazyConvTranspose2d(128, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(),
        torch.nn.LazyConvTranspose2d(64, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(), torch.nn.Upsample(scale_factor=2),
        torch.nn.LazyConvTranspose2d(32, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(), torch.nn.Upsample(scale_factor=2),
        torch.nn.LazyConvTranspose2d(16, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(), torch.nn.Upsample(scale_factor=2),
        torch.nn.LazyConv2d(1, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(),
        torch.nn.LazyLinear(87), torch.nn.Sigmoid()
    ).to(device)
    return model

def get_model2(n_predict: int, device: torch.device | str):
    model = torch.nn.Sequential(
        # torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=3 // 2),
        # torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=3 // 2),

        ResNetBlock(1, 16, 3, 1),
        ResNetBlock(16, 32, 3, 1),
        ResNetBlock(32, 64, 7, 1),
        ResNetBlock(64, 128, 7, 1),
        ResNetBlockTranspose(128, 64, 7, 1),
        ResNetBlockTranspose(64, 32, 7, 1),
        ResNetBlockTranspose(32, 16, 3, 1),
        torch.nn.Conv2d(16, 1, 3, 1, 1), torch.nn.ReLU(),
        torch.nn.LazyLinear(87), torch.nn.Sigmoid()
    ).to(device)
    return model

class DebugLayer(torch.nn.Module):
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            print(y.shape for y in x)
        else:
            print(x.shape)
        return x

class Model3(torch.nn.Module):
    def __init__(self, n_predict: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(229, 64, 16, batch_first=True, bidirectional=False)
        self.linear = torch.nn.LazyLinear(87)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # (N, 1, 128, X)
        x = x.squeeze(1)
        # (N, 128, X)
        x, _ = self.lstm(x)
        # (N, 128, 32)
        x = self.linear(x)
        # (N, 128, 87)
        x = self.sigmoid(x)
        # (N, 128, 87)
        x = x.unsqueeze(1)
        # (N, 1, 128, 87)
        return x


def get_model3(n_predict: int, device: torch.device | str):
    return Model3(n_predict).to(device)

class ModelVersion(StrEnum):
    V1 = 'v1'
    V2 = 'v2'
    V3 = 'v3'

def model_of(name: ModelVersion, n_predict: int, device: torch.device | str):
    model_getter = {
        ModelVersion.V1: get_model,
        ModelVersion.V2: get_model2,
        ModelVersion.V3: get_model3
    }.get(name, None)
    if model_getter is None:
        raise ValueError(f"Invalid model version: {name}")
    return model_getter(n_predict, device)
