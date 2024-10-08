from functools import cache
from pathlib import Path

import yaml

from src.execution_config import Config


@cache
def load_config() -> Config:
  root_path = Path(__file__).parent.parent
  config_path = root_path / 'config.yaml'
  if config_path.exists():
    # Load yaml
    data = yaml.safe_load(config_path.read_text())
    return Config.model_validate(data)
  else:
    raise FileNotFoundError(f"Config file not found at {config_path}")


def path_representer(dumper, data: Path):
  return dumper.represent_str(str(data))


config: Config = load_config()

if __name__ == '__main__':
  print("Loaded config!")
  # Dump config to yaml
  config_values = config.model_dump()
  yaml.SafeDumper.add_multi_representer(Path, path_representer)
  print(yaml.safe_dump(config_values, default_flow_style=False, default_style=''))
