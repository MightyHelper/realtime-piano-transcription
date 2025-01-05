from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, computed_field
from torch.cuda import is_available


class Config(BaseModel):
  datasets_root: Path
  maestro_name: str = 'maestro-v3.0.0'
  maestro_dataset_path: Path = Field(
    Path(maestro_name),
    description='Path to the MAESTRO dataset'
  )
  maestro_dataset_csv_path: Path = Field(
    Path(maestro_name + '.csv'),
    description='Path to the MAESTRO dataset CSV file'
  )
  maestro_duration_csv_path: Path = Field(Path("experiments") / "00_maestro_analysis" / "maestro-v3.0.0-extended.csv")
  target_sample_rate: int = 16000  # 2 ** 14
  hop_length: int = target_sample_rate * 32 // 1000
  onset_length: int = target_sample_rate * 32 // 1000
  offset_length: int = target_sample_rate * 32 // 1000
  hops_in_onset: int = onset_length // hop_length
  hops_in_offset: int = offset_length // hop_length
  min_midi: int = 21
  max_midi: int = 108
  n_mels: int = 229
  mel_fmin: int = 30
  mel_fmax: int = target_sample_rate >> 1
  window_length: int = 2048
  dummy_display: bool = False
  device: Literal['cpu', 'cuda'] = Field('cuda' if is_available() else 'cpu')

  @computed_field
  @property
  def maestro_dataset_root(self) -> Path:
    if self.maestro_dataset_path.is_absolute():
      return self.maestro_dataset_path
    return self.datasets_root / self.maestro_dataset_path

  @computed_field
  @property
  def maestro_p_dataset_root(self) -> Path:
    if self.maestro_dataset_path.is_absolute():
      return self.maestro_dataset_path
    return self.datasets_root / self.maestro_dataset_path.with_suffix(self.maestro_dataset_path.suffix + '.p')

  @computed_field
  @property
  def maestro_dataset_csv_root(self) -> Path:
    if self.maestro_dataset_csv_path.is_absolute():
      return self.maestro_dataset_csv_path
    return self.maestro_dataset_root / self.maestro_dataset_csv_path

  @computed_field
  @property
  def root(self) -> Path:
    return Path(__file__).parent.parent

  @computed_field
  @property
  def maestro_duration_csv_root(self) -> Path:
    if self.maestro_duration_csv_path.is_absolute():
      return self.maestro_duration_csv_path
    return self.root / self.maestro_duration_csv_path