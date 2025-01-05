import multiprocessing
import os
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.common import MaestroSplitType
from src.config_loader import config
from model import OnsetsAndFrames



def precompute(paths: tuple[Path, Path]) -> None:
  audio_path, tsv_path = paths
  saved_data_path = audio_path.with_suffix('.pt')
  if os.path.exists(saved_data_path):
    return None

  audio = np.load(audio_path)
  audio = torch.tensor(audio)
  audio_length = len(audio)

  n_keys = config.max_midi - config.min_midi + 1
  n_steps = (audio_length - 1) // config.hop_length + 1

  label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
  velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

  tsv_path = tsv_path
  midi = np.load(tsv_path)

  for onset, offset, note, vel in midi:
    left = int(round(onset * config.target_sample_rate / config.hop_length))
    onset_right = min(n_steps, left + config.hops_in_onset)
    frame_right = int(round(offset * config.target_sample_rate / config.hop_length))
    frame_right = min(n_steps, frame_right)
    offset_right = min(n_steps, frame_right + config.hops_in_offset)

    f = int(note) - config.min_midi
    label[left:onset_right, f] = 3
    label[onset_right:frame_right, f] = 2
    label[frame_right:offset_right, f] = 1
    velocity[left:frame_right, f] = vel

  data = dict(path=str(audio_path.absolute()), audio=audio, label=label, velocity=velocity)
  torch.save(data, saved_data_path)


def load_computed(audio_path):
  saved_data_path = audio_path.with_suffix('.pt')
  return torch.load(saved_data_path, weights_only=True)

def precompute_files(files: list[tuple[Path, Path]]) -> None:
  # multiprocessing.set_start_method('spawn')
  # torch.set_num_threads(1)
  with Pool() as p:
    list(tqdm(p.imap(precompute, files), total=len(files)))

def precompute_all_files() -> None:
  mdpr = config.maestro_p_dataset_root
  midi_files = list(mdpr.glob('**/*.midi.npy'))
  wav_files = list(mdpr.glob('**/*.wav.npy'))

  zip_files = list(zip(wav_files, midi_files))
  print(len(zip_files))

  precompute_files(zip_files)


def to_batch(data, index, sequence_length=None):
  result = dict(path=data['path'])
  random = np.random
  if sequence_length is not None:
    audio_length = len(data['audio'])
    data_length = data['label'].shape[0]
    smalest_length = min(audio_length, data_length * config.hop_length)
    step_begin = random.randint(smalest_length - sequence_length) // config.hop_length
    n_steps = sequence_length // config.hop_length
    step_end = step_begin + n_steps

    begin = step_begin * config.hop_length
    end = begin + sequence_length

    result['audio'] = data['audio'][begin:end]
    result['label'] = data['label'][step_begin:step_end, :]
    result['velocity'] = data['velocity'][step_begin:step_end, :]
  else:
    raise NotImplementedError

  assert result['audio'].min() >= -1.0
  assert result['audio'].max() <= 1.0

  result['audio'] = result['audio']
  result['onset'] = (result['label'] == 3).float()
  result['offset'] = (result['label'] == 1).float()
  result['frame'] = (result['label'] > 1).float()
  result['velocity'] = result['velocity'].float()


  return result

class MSTRODataset(Dataset):
  def __init__(self, paths: list[Path]):
      self.paths = paths

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):
    data = load_computed(self.paths[index])
    return to_batch(data, 0, config.hop_length * 320)

@contextmanager
def time_track():
  import time
  start = time.time()
  yield
  print(f"Time: {time.time() - start}")

def train(train_size=4, complexity=48, eval_size=32):
  print(f"Train {train_size} complexity {complexity}")
  mdpr = config.maestro_p_dataset_root
  wav_files = list(mdpr.glob('**/*.wav.npy'))
  learning_rate = 0.0006
  learning_rate_decay_steps = 30
  learning_rate_decay_rate = 0.98
  # multiprocessing.set_start_method('spawn')
  loader = DataLoader(MSTRODataset(wav_files), batch_size=4, shuffle=False, num_workers=1, pin_memory=config.device == 'cuda', pin_memory_device=config.device)
  model_complexity = complexity
  model = OnsetsAndFrames(config.n_mels, config.max_midi - config.min_midi + 1, model_complexity)
  model.to(config.device)
  parameters = sum(p.numel() for p in model.parameters())
  print(f"Parameters: {parameters}")
  optimizer = torch.optim.Adam(model.parameters(), learning_rate)
  scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate, verbose=True)
  lst = [*zip(loader, range(train_size))]
  lst2 = [*zip(loader, range(eval_size))]
  wandb.init(
    # set the wandb project where this run will be logged
    project="realtime-piano-transcription",

    # track hyperparameters and run metadata
    config={
      "hyper": {
        "learning_rate": learning_rate,
        "learning_rate_decay_steps": learning_rate_decay_steps,
        "learning_rate_decay_rate": learning_rate_decay_rate,
        "train_size": train_size,
        "complexity": complexity,
        "eval_size": eval_size,
        "epochs": 4,
        "model_parameters": parameters,
      },
      "device": config.device,
      "architecture": "OnsetsAndFrames",
      "dataset": {
        "name": "MAESTRO",
        "split": MaestroSplitType.TRAIN.value,
        "size": 940,
      }
    }
  )
  with prediction_display_ctx() as pd:
    for _ in range(100):
      for _ in range(100):
        with time_track():
          for batch, k in lst:
            batch = {k: v.to(config.device) if hasattr(v, 'to') else v for k, v in batch.items()}
            # Map over batch and copy to gpu
            predictions, losses = model.run_on_batch(batch)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            clip_grad_norm_(model.parameters(), 3)
            # if z % 100 == 0:
            #   print(loss.item())
            wandb.log({k: v.item() for k, v in losses.items()})
          print(">", loss.item())
      pd.send((predictions, batch))
      model.eval()
      for batch, k in lst2:
        batch = {k: v.to(config.device) if hasattr(v, 'to') else v for k, v in batch.items()}
        predictions, losses = model.run_on_batch(batch)
        loss = sum(losses.values())
        wandb.log({k + "_val": v.item() for k, v in losses.items()})
        print("Eval", loss.item())
      model.train()
  torch.save(model, f'model_train:{train_size}_compl:{complexity}.pt')
  wandb.finish()
  with open("results.txt", "a+") as f:
    f.write(f"train:{train_size}_compl:{complexity} -> {loss.item()}\n")

def prediction_display():
  with plt.ion():
    ## Using 2 subplots
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle('Predictions and Reference')
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    while True:
      prediction, batch = yield
      pred = [prediction['onset']]
      real = [batch['onset']]
      # pred = [prediction['onset'], prediction['frame']]
      # real = [batch['onset'], batch['frame']]
      axs[0].imshow(torch.cat(pred, dim=0).reshape((-1, 88)).cpu().detach().numpy().T)
      axs[0].set_title('Predictions')
      axs[1].imshow(torch.cat(real, dim=0).reshape((-1, 88)).cpu().detach().numpy().T)
      axs[1].set_title('Reference')
      axs[2].imshow(prediction['mel'].reshape((-1, 229)).cpu().detach().numpy().T)
      axs[2].set_title('Audio')
      for ax in axs.flat:
        ax.set_aspect('auto')
      fig.subplots_adjust()
      fig.tight_layout()
      fig.canvas.draw()
      fig.canvas.flush_events()

def dummy_prediction_display():
  while True:
    prediction, batch = yield

@contextmanager
def prediction_display_ctx():
  if config.dummy_display:
    pd = dummy_prediction_display()
  else:
    pd = prediction_display()
  pd.send(None)
  try:
    yield pd
  finally:
    pd.close()

def evaluate():
  # sys.path.insert(1, '/mnt/e/onsets-and-frames')
  # model = load_real_trained_model()
  model = torch.load('model_xyzz.pt').to(config.device)
  model.eval()
  with torch.no_grad():
    with prediction_display_ctx() as pd:
      mdpr = config.maestro_p_dataset_root
      wav_files = list(mdpr.glob('**/*.wav.npy'))
      # multiprocessing.set_start_method('spawn')
      loader = DataLoader(MSTRODataset(wav_files), batch_size=1, shuffle=False, pin_memory=config.device == 'cuda', pin_memory_device=config.device)
      for i, batch in zip(range(5), loader):
        print({k: v.shape if hasattr(v, 'shape') else len(v) for k, v in batch.items()})
        batch = {k: v.to(config.device) if hasattr(v, 'to') else v for k, v in batch.items()}
        prediction, loss = model.run_on_batch(batch)
        pd.send((prediction, batch))


def load_real_trained_model():
  import sys
  sys.path.insert(1, '/mnt/i/wsl_data/WinNative/GitHub/realtime-piano-transcription/src/onsets-and-frames')
  # model = torch.load('/mnt/e/onsets-and-frames/runs/transcriber-241020-234142/model-500000.pt').to(config.device)
  model = torch.load('model_xyz.pt').to(config.device)
  return model


if __name__ == '__main__':
  for n_samples in (128,):
    for model_complexity in (48,):
      if n_samples == 16 and model_complexity == 32:
        continue
      train(n_samples, model_complexity)
