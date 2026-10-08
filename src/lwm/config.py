from pathlib import Path
import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "model" not in config or "train" not in config:
        raise ValueError("Expected YAML with 'model' and 'train' sections")
    return config
