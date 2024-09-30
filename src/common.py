from logging import Logger

import librosa
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import TypedDict, Self, Generator, ClassVar
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pretty_midi import PrettyMIDI, Instrument, Note


class DatasetEntryDict(TypedDict):
    canonical_composer: str
    canonical_title: str
    split: str
    year: int
    midi_filename: str
    audio_filename: str
    duration: float
    meter: str
    tempo: float
    split_number: int


class TimeKeeper:
    TARGET_MS_PER_FRAME: ClassVar[float] = 32.

    @classmethod
    def ms_to_frames(cls, ms: float) -> int:
        return int(ms / cls.TARGET_MS_PER_FRAME)

    @classmethod
    def frames_to_ms(cls, frames: int) -> float:
        return frames * cls.TARGET_MS_PER_FRAME

    @classmethod
    def seconds_per_frame(cls) -> float:
        return cls.TARGET_MS_PER_FRAME / 1000.

    @classmethod
    def hop_size_for_rate(cls, rate: float) -> int:
        return int(rate / TimeKeeper.TARGET_MS_PER_FRAME)


@dataclass
class MidiWrapper:
    midi: PrettyMIDI
    relative_path: Path

    def get_piano_roll(self, start_seconds: float | None = None, end_seconds: float | None = None):
        if start_seconds is None:
            start_seconds = 0
        if end_seconds is None:
            end_seconds = self.duration
        return self.midi.get_piano_roll(times=np.arange(start_seconds, end_seconds, TimeKeeper.seconds_per_frame()))

    def plot_piano_roll(self):
        plt.figure(figsize=(8, 4))
        plt.imshow(self.get_piano_roll(), aspect='auto', origin='lower', cmap='gray', interpolation='none')
        plt.xlabel('Time')
        plt.ylabel('Pitch')
        plt.show()

    def save_piano_roll(self, root_path: Path) -> Path:
        """Note: Saves a y-flipped version of the piano roll"""
        path = (root_path / self.relative_path).with_suffix(".roll.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(path, self.get_piano_roll(), cmap='gray', interpolation='none')
        return path

    def plot_piano_roll_chunk(self, start_seconds: float, end_seconds: float):
        plt.figure(figsize=(int(end_seconds - start_seconds), 8))
        roll = self.get_piano_roll(start_seconds, end_seconds)
        print(roll.shape)
        plt.imshow(roll, aspect='auto', origin='lower', cmap='gray', interpolation='none')
        plt.colorbar()
        plt.xlabel('Time')
        plt.ylabel('Pitch')
        plt.show()

    def save_piano_roll_chunk(self, root_path: Path, start_seconds: float, end_seconds: float, idx: int) -> Path:
        """Note: Saves a y-flipped version of the piano roll"""
        path = (root_path / self.relative_path).with_suffix(f".{idx}.roll.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(path, self.get_piano_roll(start_seconds, end_seconds), cmap='gray')
        return path

    @property
    def duration(self):
        return self.midi.get_end_time()

    @classmethod
    def from_piano_roll(cls, piano_roll, fs=TimeKeeper.TARGET_MS_PER_FRAME, program=0, on_threshold: float = 0.5, note_offset: int = 0):
        true_false_roll = piano_roll > on_threshold
        note_events = []
        for pitch in range(true_false_roll.shape[0]):
            is_on = False
            start = 0
            for time in range(true_false_roll.shape[1]):
                if true_false_roll[pitch, time]:
                    if not is_on:
                        is_on = True
                        start = time
                else:
                    if is_on:
                        is_on = False
                        note_events.append((pitch, start, time))
            if is_on:
                note_events.append((pitch, start, true_false_roll.shape[1]))
        midi = PrettyMIDI()
        instrument = Instrument(program=program)
        for pitch, start, end in note_events:
            note = Note(
                velocity=100,
                pitch=note_offset + pitch,
                start=start / fs,
                end=end / fs
            )
            instrument.notes.append(note)
        midi.instruments.append(instrument)
        return cls(midi, Path("generated.mid"))


@dataclass
class MaestroAudio:
    rate: int
    audio: np.ndarray
    relative_path: Path | None = None
    log_mel_spectogram: np.ndarray | None = None

    def display_ipython(self, start_seconds: float = 0, end_seconds: float = None):
        from IPython.display import Audio, display
        if end_seconds is None:
            end_seconds = self.duration
        print(f"Displaying audio from {start_seconds} to {end_seconds}")
        display(Audio(
            self.audio[int(start_seconds * self.rate):int(end_seconds * self.rate)],
            rate=self.rate
        ))

    def compute_log_mel_spectrogram(self, n_mels: int = 229) -> np.ndarray:
        if self.log_mel_spectogram is not None:
            return self.log_mel_spectogram
        from librosa.feature import melspectrogram
        # rate = 16653 samples/s = 16.653 samples/ms
        # hop_length = 512 samples =
        # resolution [ms] = hop_length / rate = 512 / 16653 = 0.0307 ms
        mel = np.log(melspectrogram(y=self.audio, sr=self.rate, n_mels=n_mels,
                                    hop_length=TimeKeeper.hop_size_for_rate(self.rate)))
        self.log_mel_spectogram = mel
        return mel

    def plot_mel_spectrogram(self):
        if self.log_mel_spectogram is None:
            self.log_mel_spectogram = self.compute_log_mel_spectrogram()
        print(self.log_mel_spectogram.shape)
        plt.imshow(self.log_mel_spectogram, aspect='auto', origin='lower', cmap='gray', interpolation='none')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        # Add a color bar which maps values to colors.
        plt.colorbar(format='%+2.0f dB')
        plt.show()

    def save_raw_log_mel_spectrogram(self, root_path: Path) -> Path:
        if self.log_mel_spectogram is None:
            self.log_mel_spectogram = self.compute_log_mel_spectrogram()
        final_path = (root_path / self.relative_path).with_suffix(".mel.png")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(final_path, self.log_mel_spectogram, cmap='gray')
        return final_path

    def load_log_mel_spectrogram(self, root_path: Path):
        self.log_mel_spectogram = plt.imread(root_path / self.relative_path)

    def save(self, root_path: Path) -> Path:
        final_path = (root_path / self.relative_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(root_path / self.relative_path, self.audio)
        return final_path

    def save_audio(self, root_path: Path) -> Path:
        final_path = root_path / self.relative_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile
        soundfile.write(final_path, self.audio, self.rate)
        return final_path

    def chunk(self, start_seconds: float, end_seconds: float) -> Self:
        start_index = int(start_seconds * self.rate)
        end_index = int(end_seconds * self.rate)
        return MaestroAudio(self.rate, self.audio[start_index:end_index])

    @property
    def duration(self):
        return len(self.audio) / self.rate


@dataclass
class DatasetEntry:
    canonical_composer: str
    canonical_title: str
    split: str
    year: int
    midi_filename: str
    audio_filename: str
    duration: float

    AUDIO_RATE: ClassVar[int] = 2 ** 14
    logger: ClassVar[Logger] = Logger("DatasetEntry")

    def __post_init__(self):
        self.logger.info(f"Creating {self.audio_filename} ({id(self)})")

    @classmethod
    def from_dict(cls, d: DatasetEntryDict):
        return cls(**d)

    @cached_property
    def load_midi(self) -> MidiWrapper:
        return MidiWrapper(PrettyMIDI(str(MaestroDataset.ROOT / self.midi_filename)), Path(self.midi_filename))

    @cached_property
    def load_audio(self) -> MaestroAudio:
        """Load the associated WAV file"""
        self.logger.info(f"Loading audio {self.audio_filename} ({id(self)})")
        audio_data, rate = librosa.load(MaestroDataset.ROOT / self.audio_filename)
        # Merge stereo to mono
        if len(audio_data.shape) == 2:
            audio_data = audio_data.mean(axis=1)
        # Resample to 16KHz
        if rate != self.AUDIO_RATE:
            audio_data = librosa.resample(y=audio_data, orig_sr=rate, target_sr=self.AUDIO_RATE)
            rate = self.AUDIO_RATE
        return MaestroAudio(rate, audio_data, relative_path=Path(self.audio_filename))

    @cached_property
    def csv_duration(self) -> int:
        return MaestroDataset().duration_csv.loc[self.audio_filename].values[0]

    @cached_property
    def split_data(self) -> Generator[Path, None, None]:
        pth = MaestroDataset.SPLIT_ROOT / self.split / self.audio_filename
        print(pth)
        print(pth.stem)
        for path in pth.parent.iterdir():
            if pth.stem in path.name:
                yield path

    @property
    def split_file_types(self) -> tuple[list[Path], list[Path], list[Path]]:
        roll_files = []
        mel_files = []
        audio_files = []

        for path in self.split_data:
            if path.suffix == ".roll.png":
                roll_files.append(path)
            elif path.suffix == ".mel.png":
                mel_files.append(path)
            else:
                audio_files.append(path)
        return roll_files, mel_files, audio_files


class MaestroSplitType(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class MaestroSplit:
    split_type: MaestroSplitType
    entries: list[DatasetEntry]

    def __post_init__(self):
        self.df_entries = pd.DataFrame([entry.__dict__ for entry in self.entries])

    def load_index(self, index):
        return self.entries[index]


class MaestroDataset:
    ROOT = Path("/mnt") / "e" / "datasets" / "maestro" / "maestro-v3.0.0"
    SPLIT_ROOT = Path("/mnt") / "e" / "datasets" / "maestro" / "maestro-v3.0.0-split"
    CSV = ROOT / "maestro-v3.0.0.csv"
    DURATION_CSV = Path(__file__).parent.parent / "experiments" / "00_maestro_analysis" / "maestro-v3.0.0-extended.csv"

    @cached_property
    def csv(self):
        return pd.read_csv(self.CSV)

    @cached_property
    def duration_csv(self):
        return pd.read_csv(self.DURATION_CSV, index_col=0)

    def length(self):
        return len(self.csv)

    def get_split(self, split_type: MaestroSplitType) -> MaestroSplit:
        return MaestroSplit(
            split_type,
            [
                DatasetEntry.from_dict(DatasetEntryDict(**row))
                for index, row in self.csv[self.csv["split"] == split_type].iterrows()
            ]
        )

    def load_index(self, index: int) -> DatasetEntry:
        return DatasetEntry.from_dict(DatasetEntryDict(**self.csv.iloc[index]))

    def get_real_duration(self, index: int) -> float:
        # return self.duration_csv[self.duration_csv.index == self.csv.iloc[index]['audio_filename']].iloc[0].values[0]
        return self.duration_csv.loc[self.csv.iloc[index]['audio_filename']].values[0]
