import torch
from torch import nn
import torchaudio


def vd(t: torch.Tensor, d: tuple[int, ...]) -> torch.Tensor:
    """Validate the shape of a tensor."""
    assert t.shape == torch.Size(d), f"Expected shape {d}, got {t.shape}"
    return t


class FrequencyConditionedFiLM(nn.Module):
    def __init__(self, num_channels, frequency_bins, hidden_size=16):
        """
        Args:
            num_channels (int): Number of channels in the input feature map.
            frequency_bins (int): Number of bins along the frequency axis (F).
            hidden_size (int): Size of the middle layers in gamma and beta networks.
        """
        super(FrequencyConditionedFiLM, self).__init__()
        self.frequency_bins = frequency_bins

        # Fully connected layers for gamma and beta
        self.gamma_fc = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_channels)
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_channels)
        )

    def forward(self, feature_map):
        """
        Args:
            feature_map (torch.Tensor): Input feature map (B, C, F, T).
        Returns:
            torch.Tensor: Modulated feature map.
        """
        B, C, F, T = feature_map.shape
        vd(feature_map, (B, C, F, T))  # Validate input shape

        # Generate relative frequency height k/F as (F, 1)
        relative_frequency = torch.arange(F, device=feature_map.device).float() / F
        relative_frequency = relative_frequency.view(F, 1)  # Shape (F, 1)
        vd(relative_frequency, (F, 1))

        # Compute gamma and beta
        gamma = self.gamma_fc(relative_frequency).unsqueeze(0).unsqueeze(-1)  # Shape (1, F, C, 1)
        vd(gamma, (1, F, C, 1))
        beta = self.beta_fc(relative_frequency).unsqueeze(0).unsqueeze(-1)  # Shape (1, F, C, 1)
        vd(beta, (1, F, C, 1))

        # Apply FiLM modulation
        feature_map = feature_map.permute(0, 2, 1, 3)  # (B, F, C, T)
        vd(feature_map, (B, F, C, T))
        modulated = gamma * feature_map + beta
        vd(modulated, (B, F, C, T))
        modulated = modulated.permute(0, 2, 1, 3)  # Back to (B, C, F, T)
        vd(modulated, (B, C, F, T))

        return modulated


class ConvFiLMBlock(nn.Module):
    def __init__(self, input_channels, output_channels, frequency_bins, kernel_size=3, stride=1, padding=1,
                 hidden_size=16):
        """
        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            frequency_bins (int): Number of bins along the frequency axis (F).
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding for the convolution.
            dropout (float): Dropout rate.
            hidden_size (int): Size of the middle layers in gamma and beta networks of FiLM.
        """
        super(ConvFiLMBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        # FiLM modulation
        self.film = FrequencyConditionedFiLM(output_channels, frequency_bins, hidden_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (B, C_in, F, T).
        Returns:
            torch.Tensor: Processed tensor (B, C_out, F, T).
        """
        B, C, F, T = x.shape
        vd(x, (B, C, F, T))  # Validate input shape
        x1 = vd(self.conv1(x), (B, self.conv1.out_channels, F, T))  # CNN + batch norm
        x2 = vd(self.conv2(x1), (B, self.conv2.out_channels, F, T))
        x2 = self.batch_norm(x2)
        x_film = self.film(x2)  # FiLM modulation
        x_film = vd(x_film + x, (B, self.conv2.out_channels, F, T))
        return x_film


class AcousticModule(nn.Module):
    def __init__(self,
                 sample_rate,
                 n_fft,
                 n_mels,
                 hop_length,
                 input_channels,
                 conv_channels,
                 num_blocks,
                 hidden_size=16,
                 dropout=0.25,
                 timewise_fc_size=768,
                 pitchwise_split_size=88):
        """
        Args:
            sample_rate (int): Sample rate of the audio.
            n_fft (int): FFT window size for spectrogram computation.
            n_mels (int): Number of mel frequency bins.
            hop_length (int): Hop length for STFT.
            input_channels (int): Number of input channels (e.g., spectrogram channels).
            conv_channels (list[int]): List specifying the number of channels for each ConvFiLM Block.
            num_blocks (int): Number of ConvFiLM Blocks.
            hidden_size (int): Size of the hidden layers in gamma and beta networks of FiLM.
            dropout (float): Dropout rate applied to the module.
            timewise_fc_size (int): Number of units in the timewise fully connected layer.
            pitchwise_split_size (int): Number of pitch segments (e.g., 88 for piano keys).
        """
        super(AcousticModule, self).__init__()

        assert len(conv_channels) == num_blocks, "Number of blocks must match the length of conv_channels."

        # Spectrogram layer (Mel)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # ConvFiLM Blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                ConvFiLMBlock(
                    input_channels=input_channels if i == 0 else conv_channels[i - 1],
                    output_channels=conv_channels[i],
                    frequency_bins=n_mels // (2 ** i),  # Frequency bins reduce after each max-pool
                    hidden_size=hidden_size
                )
            )

        # Max-pooling to reduce frequency dimension
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 1))  # Only downsample frequency dimension

        # Dropout
        self.dropout = nn.Dropout2d(dropout)

        # Timewise FCs
        self.timewise_fc1 = nn.Linear(conv_channels[-1] * (n_mels // (2 ** num_blocks)), timewise_fc_size)
        self.timewise_fc2 = nn.Linear(timewise_fc_size, pitchwise_split_size * timewise_fc_size)

        # Pitchwise split
        self.pitchwise_split_size = pitchwise_split_size
        self.timewise_fc_size = timewise_fc_size

    def forward(self, audio):
        """
        Args:
            audio (torch.Tensor): Input raw audio tensor (B, T).
        Returns:
            torch.Tensor: Output tensor after feature extraction (B, 88, FC_size, T_out).
        """
        # Compute mel spectrogram
        mel_spec = self.mel_spectrogram(audio)  # Shape (B, 1, F, T)
        mel_spec = self.db_transform(mel_spec)  # Convert to dB
        vd(mel_spec, (audio.shape[0], 1, self.mel_spectrogram.n_mels, -1))  # Validate spectrogram shape

        x = mel_spec
        for block in self.blocks:
            x = block(x)  # Pass through ConvFiLM Block
            x = self.max_pool(x)  # Reduce frequency dimension
            x = self.dropout(x)  # Apply dropout

        B, C, F, T = x.shape
        vd(x, (audio.shape[0], self.blocks[-1].conv2.out_channels, F, T))

        # Reshape for Timewise FCs
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x = x.reshape(B, T, -1)  # Flatten to (B, T, C * F)
        vd(x, (B, T, C * F))

        x = self.timewise_fc1(x)  # Timewise fully connected layer 1
        vd(x, (B, T, self.timewise_fc_size))

        x = self.timewise_fc2(x)  # Timewise fully connected layer 2
        vd(x, (B, T, self.pitchwise_split_size * self.timewise_fc_size))

        # Reshape for pitchwise split
        x = x.view(B, T, self.pitchwise_split_size, self.timewise_fc_size)  # (B, T, 88, FC_size)
        x = x.permute(0, 2, 3, 1)  # (B, 88, FC_size, T)
        vd(x, (B, self.pitchwise_split_size, self.timewise_fc_size, T))

        return x
