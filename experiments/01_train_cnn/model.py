import torch

def get_model(n_predict: int, device: torch.device | str):
    model = torch.nn.Sequential(
        torch.nn.LazyConv2d(16, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.LazyConv2d(32, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.LazyConv2d(64, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.LazyConv2d(128, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(),
        torch.nn.LazyConv2d(128, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(),
        torch.nn.LazyConvTranspose2d(128, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(),
        torch.nn.LazyConvTranspose2d(128, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(),
        torch.nn.LazyConvTranspose2d(64, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(), torch.nn.Upsample(scale_factor=2),
        torch.nn.LazyConvTranspose2d(32, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(), torch.nn.Upsample(scale_factor=2),
        torch.nn.LazyConvTranspose2d(16, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(), torch.nn.Upsample(scale_factor=2),
        torch.nn.LazyConv2d(1, kernel_size=(7, 3), padding=(3, 1)), torch.nn.ReLU(),
        torch.nn.LazyLinear(87), torch.nn.Sigmoid()
    ).to(device)
    return model

