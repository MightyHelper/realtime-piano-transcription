from functools import cache
from pathlib import Path
from multiprocessing import Pool
import librosa
import numpy as np
from tqdm.auto import tqdm
from midi import parse_midi
from src.config_loader import config


@cache
def self_modification_time() -> float:
  return Path(__file__).stat().st_mtime

def convert_single_midi_file(files: tuple[Path, Path]) -> None:
  file, target_file = files
  if target_file.exists() and target_file.stat().st_mtime > self_modification_time():
    return
  notes = parse_midi(file)
  target_file.parent.mkdir(parents=True, exist_ok=True)
  np.save(target_file, notes)

def convert_midi_files(files: list[Path], target_files: list[Path]) -> None:
  """
  Take files[x] and convert them to npy files in target_files[x]
  :param files: The files to convert
  :param target_files: The target files
  :return:
  """
  with Pool() as p:
    list(tqdm(p.imap(convert_single_midi_file, zip(files, target_files)), total=len(files)))

def convert_single_audio_file_librosa(files: tuple[Path, Path]) -> None:
  file, target_file = files
  if target_file.exists() and target_file.stat().st_mtime > self_modification_time():
    return
  audio, sr = librosa.load(file, sr=None)
  if len(audio.shape) >= 2:
    audio = audio.mean(axis=1)
  if sr != config.target_sample_rate:
    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=config.target_sample_rate)
  target_file.parent.mkdir(parents=True, exist_ok=True)
  np.save(target_file, audio)

def convert_audio_files(files: list[Path], target_files: list[Path]) -> None:
  """
  Take files[x] and convert them to npy files in target_files[x]
  :param files: The files to convert
  :param target_files: The target files
  :return:
  """
  with Pool(8) as p:
    list(tqdm(p.imap(convert_single_audio_file_librosa, zip(files, target_files)), total=len(files)))

def main():
  mdr = config.maestro_dataset_root
  mdpr = config.maestro_p_dataset_root
  midi_files = list(mdr.glob('**/*.midi'))
  wav_files = list(mdr.glob('**/*.wav'))
  print(len(list(midi_files)), len(list(wav_files)))
  # convert_midi_files(midi_files, [mdpr / file.relative_to(mdr).with_suffix('.midi.npy') for file in midi_files])
  convert_audio_files(wav_files, [mdpr / file.relative_to(mdr).with_suffix('.wav.npy') for file in wav_files])

if __name__ == '__main__':
  main()

