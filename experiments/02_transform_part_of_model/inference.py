import multiprocessing
import os
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tensorboard.summary.v1 import audio
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

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
  multiprocessing.set_start_method('spawn')
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
  # data = data[index:, :, :]
  result = dict(path=data['path'])
  random = np.random
  device = config.device
  if sequence_length is not None:
    audio_length = len(data['audio'])
    data_length = data['label'].shape[0]
    smalest_length = min(audio_length, data_length * config.hop_length)
    step_begin = random.randint(smalest_length - sequence_length) // config.hop_length
    n_steps = sequence_length // config.hop_length
    step_end = step_begin + n_steps

    begin = step_begin * config.hop_length
    end = begin + sequence_length

    print(f"{begin} {end}")
    print(f"{step_begin} {step_end}")
    print(f"{audio_length} {data_length} {smalest_length}")
    print(f"{data['audio'].shape}")
    print(f"{data['label'].shape}")
    print(f"{data['velocity'].shape}")
    result['audio'] = data['audio'][begin:end].to(device)
    result['label'] = data['label'][step_begin:step_end, :].to(device)
    result['velocity'] = data['velocity'][step_begin:step_end, :].to(device)
  else:
    raise NotImplementedError
    # result['audio'] = data['audio'].to(device)
    # result['label'] = data['label'].to(device)
    # result['velocity'] = data['velocity'].to(device).float()

  # Assert result['audio'] values are floats between -1 and 1

  assert result['audio'].min() >= -1.0
  assert result['audio'].max() <= 1.0

  result['audio'] = result['audio']
  result['onset'] = (result['label'] == 3).float()
  result['offset'] = (result['label'] == 1).float()
  result['frame'] = (result['label'] > 1).float()
  result['velocity'] = result['velocity'].float()

  # Print min and max values for each tensor
  print('audio', result['audio'].min().cpu().detach().item(), result['audio'].max().cpu().detach().item())
  print('label', result['label'].min().cpu().detach().item(), result['label'].max().cpu().detach().item())
  print('velocity', result['velocity'].min().cpu().detach().item(), result['velocity'].max().cpu().detach().item())
  print('onset', result['onset'].min().cpu().detach().item(), result['onset'].max().cpu().detach().item())
  print('offset', result['offset'].min().cpu().detach().item(), result['offset'].max().cpu().detach().item())
  print('frame', result['frame'].min().cpu().detach().item(), result['frame'].max().cpu().detach().item())


  return result

class MSTRODataset(Dataset):
  def __init__(self, paths: list[Path]):
    self.paths = paths

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):
    data = load_computed(self.paths[index])
    return to_batch(data, 0, config.hop_length *900)

def train(num_iter: int = 2050):
  mdpr = config.maestro_p_dataset_root
  wav_files = list(mdpr.glob('**/*.wav.npy'))
  learning_rate = 0.0006
  learning_rate_decay_steps = 10000
  learning_rate_decay_rate = 0.98
  multiprocessing.set_start_method('spawn')
  loader = DataLoader(MSTRODataset(wav_files), batch_size=2, shuffle=True)#, num_workers=4)
  model_complexity = 48
  model = OnsetsAndFrames(config.n_mels, config.max_midi - config.min_midi + 1, model_complexity)
  model.to(config.device)
  print(model)
  print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
  optimizer = torch.optim.Adam(model.parameters(), learning_rate)
  scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate, verbose=True)
  with prediction_display_ctx() as pd:
    for batch,_ in zip(loader, range(1)):
      for k in range(1000):
        accumulated_losses = {
          'loss/onset': 0,
          'loss/offset': 0,
          'loss/frame': 0,
          'loss/velocity': 0,
          'loss': 0
        }
        prediction, loss = model.run_on_batch(batch)
        optimizer.zero_grad()
        loss['loss/onset'].backward()
        # loss['loss'].backward()
        optimizer.step()
        scheduler.step()
        clip_grad_norm_(model.parameters(), 3)
        pd.send((prediction, batch))
        loss['loss'] = sum(loss.values())
        accumulated_losses = {x: accumulated_losses[x] + loss[x].cpu().detach().item() for x in accumulated_losses}
        print(prediction['onset'].shape)
        # Print std along time axis
        print(prediction['onset'].min().cpu().detach().item(), prediction['onset'].max().cpu().detach().item())
        # Print std along batch axis
        torch.save(model, 'model.pt')
        print(' '.join(f"{x}: {lz:.4f}" for x, lz in accumulated_losses.items()))

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
  model = torch.load('model.pt').to(config.device)
  model.eval()
  with torch.no_grad():
    with prediction_display_ctx() as pd:
      mdpr = config.maestro_p_dataset_root
      wav_files = list(mdpr.glob('**/*.wav.npy'))
      # multiprocessing.set_start_method('spawn')
      loader = DataLoader(MSTRODataset(wav_files), batch_size=1, shuffle=True)
      for i, batch in zip(range(5), loader):
        print({k: v.shape if hasattr(v, 'shape') else len(v) for k, v in batch.items()})
        prediction, loss = model.run_on_batch(batch)
        pd.send((prediction, batch))


def load_real_trained_model():
  import sys
  sys.path.insert(1, '/mnt/i/wsl_data/WinNative/GitHub/realtime-piano-transcription/src/onsets-and-frames')
  model = torch.load('/mnt/e/onsets-and-frames/runs/transcriber-241020-234142/model-500000.pt').to(config.device)
  return model


if __name__ == '__main__':
  train()