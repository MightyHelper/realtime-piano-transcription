import json
from pathlib import Path
from typing import TypedDict
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

from src.common import MaestroSplitType, MaestroDataset, dataset


class ProcessedDatasetEntry(TypedDict):
    audio_path: str
    midi_path: str
    mel_path: str
    original_audio_path: str
    original_midi_path: str

class MaestroDatasetSplit(Dataset):
    split_type: MaestroSplitType
    metadata_path: Path
    metadata: list[ProcessedDatasetEntry]

    def __init__(self, split_type: MaestroSplitType):
        self.split_type = split_type
        self.metadata_path = MaestroDataset.SPLIT_ROOT / split_type.lower() / "metadata.json"
        with self.metadata_path.open() as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata[idx]
        mel_path = MaestroDataset.SPLIT_ROOT / Path(sample["mel_path"])
        roll_path = MaestroDataset.SPLIT_ROOT / Path(sample["midi_path"])

        mel_image = plt.imread(mel_path)
        roll_image = plt.imread(roll_path)

        mel_tensor = torch.tensor(mel_image)
        roll_tensor = torch.tensor(roll_image)

        return mel_tensor[:,:,0], roll_tensor[:,:,0]



if __name__ == "__main__":
    print((((dataset.csv[dataset.csv['split'] == 'train'])['duration'] // 10) + 0.5).round().sum())
