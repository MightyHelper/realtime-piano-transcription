import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def assert_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], layer_name: str):
    if list(tensor.size()) != list(expected_shape):
        raise RuntimeError(f"[{layer_name}] Tensor shape {tensor.shape} does not match expected {expected_shape}")

class FiLM(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.gamma_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_channels)
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        freq_positions = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1)
        gamma = self.gamma_fc(freq_positions).view(1, 1, H, C).permute(0, 3, 2, 1).expand(B, C, H, W)
        beta = self.beta_fc(freq_positions).view(1, 1, H, C).permute(0, 3, 2, 1).expand(B, C, H, W)
        output = gamma * x + beta
        assert_shape(output, (B, C, H, W), "FiLM")
        return output

class ConvFiLMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.film = FiLM(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        x = self.conv(x)
        assert_shape(x, (B, self.conv.out_channels, H, W), "ConvFiLMBlock: After Conv")
        x = self.bn(x)
        x = F.relu(self.film(x))
        assert_shape(x, (B, self.conv.out_channels, H, W), "ConvFiLMBlock: Output")
        return x

class AcousticModule(nn.Module):
    def __init__(self, hidden_units_per_pitch: int = 8):
        super().__init__()
        self.conv1 = ConvFiLMBlock(1, 48)
        self.conv2 = ConvFiLMBlock(48, 48)
        self.conv3 = ConvFiLMBlock(48, 48)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.fc1 = nn.Linear(48 * 175, 768)
        self.fc2 = nn.Linear(768, 88 * hidden_units_per_pitch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        x = self.conv1(x)
        assert_shape(x, (B, 48, H, W), "AcousticModule: After Conv1")
        x = self.pool(x)
        assert_shape(x, (B, 48, H // 2, W), "AcousticModule: After Pool1")
        x = self.conv2(x)
        assert_shape(x, (B, 48, H // 2, W), "AcousticModule: After Conv2")
        x = self.pool(x)
        assert_shape(x, (B, 48, H // 4, W), "AcousticModule: After Pool2")
        x = self.conv3(x)
        assert_shape(x, (B, 48, H // 4, W), "AcousticModule: After Conv3")
        x = x.permute(0, 3, 2, 1)
        assert_shape(x, (B, W, H // 4, 48), "AcousticModule: After Permute")
        x = x.flatten(2)
        assert_shape(x, (B, W, 48 * (H // 4)), "AcousticModule: After Flatten")
        x = self.fc1(x)
        assert_shape(x, (B, W, 768), "AcousticModule: After FC1")
        x = self.fc2(x)
        assert_shape(x, (B, W, 88 * (x.size(-1) // 88)), "AcousticModule: Output")
        return x

class PitchwiseLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, T, D = x.size()
        outputs = []
        for p in range(P):
            pitch_features = x[:, p, :, :]
            assert_shape(pitch_features, (B, T, D), f"PitchwiseLSTM: Pitch {p} Input")
            lstm_out, _ = self.lstm(pitch_features)
            assert_shape(lstm_out, (B, T, self.lstm.hidden_size), f"PitchwiseLSTM: Pitch {p} LSTM Output")
            outputs.append(self.fc(lstm_out))
        output = torch.stack(outputs, dim=1)
        assert_shape(output, (B, P, T, 5), "PitchwiseLSTM: Output")
        return output

class PARModel(nn.Module):
    def __init__(self, hidden_units_per_pitch: int = 8):
        super().__init__()
        self.acoustic_module = AcousticModule(hidden_units_per_pitch=hidden_units_per_pitch)
        self.note_sequence_module = PitchwiseLSTM(input_dim=hidden_units_per_pitch, hidden_dim=48, output_dim=5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        assert_shape(x, (B, 1, H, W), "PARModel: Input")
        acoustic_features = self.acoustic_module(x)
        assert_shape(acoustic_features, (B, W, 88 * (acoustic_features.size(-1) // 88)), "PARModel: Acoustic Features")
        T = acoustic_features.size(1)
        acoustic_features = acoustic_features.view(B, T, 88, acoustic_features.size(-1) // 88).permute(0, 2, 1, 3)
        assert_shape(acoustic_features, (B, 88, T, acoustic_features.size(-1) // 88), "PARModel: Reshaped Acoustic Features")
        note_states = self.note_sequence_module(acoustic_features)
        assert_shape(note_states, (B, 88, T, 5), "PARModel: Output")
        return note_states

if __name__ == "__main__":
    batch_size = 8
    time_steps = 200
    mel_bins = 700
    input_tensor = torch.rand((batch_size, 1, mel_bins, time_steps))
    par_model = PARModel()
    output = par_model(input_tensor)
    print(f"Final Output shape: {output.shape}")
