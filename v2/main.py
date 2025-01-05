import torch
import torch.nn as nn
import torchaudio.transforms as transforms

class FiLM(nn.Module):
    def __init__(self, num_channels, freq_bins):
        super(FiLM, self).__init__()
        self.gamma_fc = nn.Sequential(
            nn.Linear(freq_bins, 16), nn.ReLU(),
            nn.Linear(16, num_channels)
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(freq_bins, 16), nn.ReLU(),
            nn.Linear(16, num_channels)
        )

    def forward(self, x, freq_condition):
        # Input:
        #   x: (batch, num_channels, height, width)
        #   freq_condition: (batch, freq_bins)
        # Output: (batch, num_channels, height, width)
        freq_condition = freq_condition / freq_condition.max(dim=-1, keepdim=True).values
        gamma = self.gamma_fc(freq_condition).unsqueeze(-1).unsqueeze(-1)  # (batch, num_channels, 1, 1)
        beta = self.beta_fc(freq_condition).unsqueeze(-1).unsqueeze(-1)    # (batch, num_channels, 1, 1)
        return gamma * x + beta

class ConvFiLMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, freq_bins):
        super(ConvFiLMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # (batch, out_channels, height, width)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # (batch, out_channels, height, width)
        self.bn = nn.BatchNorm2d(out_channels)  # (batch, out_channels, height, width)
        self.film = FiLM(out_channels, freq_bins)
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))  # (batch, out_channels, height // 2, width)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else None

    def forward(self, x, freq_condition):
        # Input:
        #   x: (batch, in_channels, height, width)
        #   freq_condition: (batch, freq_bins)
        # Output: (batch, out_channels, height // 2, width)
        residual = self.residual(x) if self.residual else x  # (batch, out_channels, height, width)
        x = self.conv1(x)  # (batch, out_channels, height, width)
        x = self.conv2(x)  # (batch, out_channels, height, width)
        x = self.bn(x)  # (batch, out_channels, height, width)
        x = self.film(x, freq_condition)  # (batch, out_channels, height, width)
        x = x + residual  # (batch, out_channels, height, width)
        x = self.dropout(x)  # (batch, out_channels, height, width)
        x = self.pool(x)  # (batch, out_channels, height // 2, width)
        return x

class AcousticModule(nn.Module):
    def __init__(self, freq_bins=700, num_channels=48, num_pitches=88):
        super(AcousticModule, self).__init__()
        self.log_mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=16000, n_mels=freq_bins, n_fft=2048, hop_length=512
        )
        self.conv_film1 = ConvFiLMBlock(1, num_channels, freq_bins)
        self.conv_film2 = ConvFiLMBlock(num_channels, num_channels, freq_bins // 2)
        self.conv_film3 = ConvFiLMBlock(num_channels, num_channels, freq_bins // 4)
        self.reduce_channels = nn.Conv2d(num_channels, 1, kernel_size=1)  # (batch, 1, height, width)
        self.fc = nn.Linear(freq_bins // 4, num_channels)  # (batch, num_channels)

    def forward(self, x, freq_condition):
        # Input:
        #   x: (batch, time_samples)
        #   freq_condition: (batch, freq_bins)
        # Output: (batch, num_channels)
        x = self.log_mel_spectrogram(x).unsqueeze(1)  # (batch, 1, freq_bins, time)
        x = self.conv_film1(x, freq_condition)  # (batch, num_channels, freq_bins // 2, time)
        x = self.conv_film2(x, freq_condition)  # (batch, num_channels, freq_bins // 4, time)
        x = self.conv_film3(x, freq_condition)  # (batch, num_channels, freq_bins // 8, time)
        x = self.reduce_channels(x).squeeze(1)  # (batch, freq_bins // 8, time)
        x = self.fc(x.permute(0, 2, 1))  # (batch, time, num_channels)
        return x

class PitchwiseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_pitches=88):
        super(PitchwiseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # (batch * num_pitches, time, hidden_size)
        self.fc = nn.Linear(hidden_size, 5)  # (batch * num_pitches, time, 5)

    def forward(self, x):
        # Input:
        #   x: (batch, num_pitches, time, input_size)
        # Output: (batch, num_pitches, time, 5)
        batch, num_pitches, time, input_size = x.size()
        x = x.view(batch * num_pitches, time, input_size)  # Flatten pitch dimension
        x, _ = self.lstm(x)  # (batch * num_pitches, time, hidden_size)
        x = self.fc(x)  # (batch * num_pitches, time, 5)
        x = x.view(batch, num_pitches, time, 5)  # Reshape to original
        return x

class PARCompact(nn.Module):
    def __init__(self, freq_bins=700, num_channels=48, lstm_hidden_size=48, num_pitches=88):
        super(PARCompact, self).__init__()
        self.acoustic_module = AcousticModule(freq_bins, num_channels, num_pitches)
        self.pitchwise_lstm = PitchwiseLSTM(num_channels, lstm_hidden_size, num_pitches)

    def forward(self, waveform, freq_condition, recursive_context):
        # Input:
        #   waveform: (batch, time_samples)
        #   freq_condition: (batch, freq_bins)
        #   recursive_context: (batch, num_pitches, time, context_dim)
        # Output: (batch, num_pitches, time, 5)
        acoustic_features = self.acoustic_module(waveform, freq_condition)  # (batch, time, num_channels)
        acoustic_features = acoustic_features.unsqueeze(1).expand(-1, 88, -1, -1)  # (batch, num_pitches, time, num_channels)
        input_features = torch.cat([acoustic_features, recursive_context], dim=-1)  # (batch, num_pitches, time, num_channels + context_dim)
        output = self.pitchwise_lstm(input_features)  # (batch, num_pitches, time, 5)
        return output

class RecursiveContext(nn.Module):
    def __init__(self, input_dim, output_dim=4):
        super(RecursiveContext, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, note_state, duration, velocity):
        # Input:
        #   note_state, duration, velocity: (batch, num_pitches, time, input_dim / 3)
        # Output: (batch, num_pitches, time, output_dim)
        context = torch.cat([note_state, duration, velocity], dim=-1)  # (batch, num_pitches, time, input_dim)
        return self.fc(context)  # (batch, num_pitches, time, output_dim)

