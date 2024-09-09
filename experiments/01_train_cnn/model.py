import torch

def get_model(n_predict: int, device: torch.device | str):
    model = torch.nn.Sequential(
        torch.nn.LazyConv2d(16, kernel_size=3, padding=1), torch.nn.ReLU(),
        torch.nn.LazyConv2d(32, kernel_size=3, padding=1), torch.nn.ReLU(),
        torch.nn.LazyConv2d(64, kernel_size=3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.LazyConv2d(128, kernel_size=3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.LazyConv2d(128, kernel_size=3, padding=1), torch.nn.ReLU(),
        torch.nn.LazyConv2d(128, kernel_size=3, padding=1), torch.nn.ReLU(),
        torch.nn.LazyConv2d(128, kernel_size=3, padding=1), torch.nn.ReLU(),
        torch.nn.LazyConv2d(128, kernel_size=3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        # Flatten
        torch.nn.Flatten(),
        torch.nn.LazyLinear(16 * 128 * n_predict),
        torch.nn.LazyLinear(128 * n_predict),
        torch.nn.Sigmoid(),
        # Unflatten to (batch, 128, n_predict)
        torch.nn.Unflatten(1, (128, n_predict)),
    ).to(device)
    return model

