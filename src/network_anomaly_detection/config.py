import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file {config_path} is empty")
    
    return config
