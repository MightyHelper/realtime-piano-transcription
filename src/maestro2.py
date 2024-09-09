from torch import tensor, stack, cat
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from src.common import MaestroSplitType, MaestroDataset, MaestroSplit, MaestroAudio, MidiWrapper, TimeKeeper
from functools import cache
from textwrap import dedent


class MaestroDatasetSplit(Dataset):
    split_type: MaestroSplitType
    split: MaestroSplit

    def __init__(self, split_type: MaestroSplitType):
        super().__init__()
        self.split_type = split_type
        self.base_dataset = MaestroDataset()
        self.split = self.base_dataset.get_split(split_type)

    @cache
    def __len__(self):
        return len(self.split.df_entries)

    def __getitem__(self, index: int):
        item = self.split.load_index(index)
        audio: MaestroAudio = item.load_audio
        midi: MidiWrapper = item.load_midi
        mel = audio.compute_log_mel_spectrogram()
        roll = midi.get_piano_roll()

        mel_duration_frames = mel.shape[1]
        roll_duration_frames = roll.shape[1]

        min_duration_frames = min(mel_duration_frames, roll_duration_frames)
        max_duration_frames = max(mel_duration_frames, roll_duration_frames)

        mel = mel[:, :min_duration_frames]
        roll = roll[:, :min_duration_frames]

        # If the difference is more than 5 seconds
        if abs(mel_duration_frames - roll_duration_frames) > 0.1 * max_duration_frames:
            text = dedent(f"""
            Mel : {mel.shape[1]} ({mel_duration_frames})
            roll: {roll.shape[1]} ({roll_duration_frames})
            item: {item.duration * TimeKeeper.TARGET_MS_PER_FRAME}
            min : {min_duration_frames}
            max : {max_duration_frames}
            csv : {item.csv_duration}
            """)

            raise ValueError(
                f"Mel and roll duration mismatch by {abs(mel_duration_frames - roll_duration_frames)}"
                f" frames ({TimeKeeper.frames_to_ms(abs(mel_duration_frames - roll_duration_frames))}ms)" + text
            )

        return tensor(mel).float(), tensor(roll).float()

    def total_frame_count(self):
        return sum([entry.csv_duration for entry in self.split.entries])


class FrameContextDataset(Dataset):
    def __init__(self, dataset: MaestroDatasetSplit, context_frames: int, predict_frames: int):
        self.dataset = dataset
        self.context_frames = context_frames
        self.predict_frames = predict_frames

    @cache
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # mel: (mel_bins, frames)
        # roll: (roll_keys, frames)
        mel, roll = self.dataset[index]
        # batch_size = frames - context_frames
        batch_size = mel.shape[1] - self.context_frames
        print(f"Batch size: {batch_size}")
        # (batch_size, mel_bins, context_frames)
        mel_context = stack([mel[:, i:(i + self.context_frames)] for i in range(batch_size)])
        # (batch_size, roll_keys, predict_frames)
        roll_context = stack(
            [roll[:, i + self.context_frames - self.predict_frames:i + self.context_frames] for i in range(batch_size)]
        )
        return mel_context, roll_context

    def total_frame_count(self):
        return self.dataset.total_frame_count()


class DynamicBatchIterableDataset(IterableDataset):
    def __init__(self, original_dataset: FrameContextDataset, standard_batch_size: int):
        self.original_dataset = original_dataset
        self.standard_batch_size = standard_batch_size

    def __iter__(self):
        # Iterate over the original dataset
        for X, Y in self.original_dataset:
            start_idx = 0
            # Slice smaller batches from the larger batch
            while start_idx < X.shape[0]:
                end_idx = min(start_idx + self.standard_batch_size, X.shape[0])
                x_batch = X[start_idx:end_idx]
                y_batch = Y[start_idx:end_idx]
                start_idx = end_idx
                yield x_batch, y_batch

    def __len__(self):
        return self.original_dataset.total_frame_count() // self.standard_batch_size


class DynamicBatchIterableDataset2(IterableDataset):
    def __init__(self, original_dataset, standard_batch_size):
        self.original_dataset = original_dataset
        self.standard_batch_size = standard_batch_size

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # Single-process data loading
            dataset_iter = iter(self.original_dataset)
        else:  # In multi-process loading, split the workload
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            total_data_length = len(self.original_dataset)

            # Divide the dataset indices among workers
            per_worker = total_data_length // num_workers
            remainder = total_data_length % num_workers

            # Calculate the start and end indices for this worker
            start_idx = worker_id * per_worker + min(worker_id, remainder)
            end_idx = start_idx + per_worker + (1 if worker_id < remainder else 0)

            # Create an iterator over the assigned chunk
            dataset_iter = range(start_idx, end_idx)

        for i in dataset_iter:
            X, Y = self.original_dataset[i]
            start_idx = 0
            # Slice smaller batches from the larger batch
            while start_idx < X.shape[0]:
                end_idx = min(start_idx + self.standard_batch_size, X.shape[0])
                X_batch = X[start_idx:end_idx]
                Y_batch = Y[start_idx:end_idx]
                start_idx = end_idx
                yield X_batch, Y_batch

    def __len__(self):
        return self.original_dataset.total_frame_count() // self.standard_batch_size

def custom_collate_fn(batch):
    x_list, y_list = zip(*batch)
    x_batch = cat(x_list, dim=0)
    y_batch = cat(y_list, dim=0)
    return x_batch, y_batch
