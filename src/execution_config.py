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
  device: Literal['cpu', 'cuda'] = Field('cuda' if is_available() else 'cpu')

  @computed_field
  @property
  def maestro_dataset_root(self) -> Path:
    if self.maestro_dataset_path.is_absolute():
      return self.maestro_dataset_path
    return self.datasets_root / self.maestro_dataset_path

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