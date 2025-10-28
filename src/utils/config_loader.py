"""YAML + .env config loader."""
import yaml
import os
from dotenv import load_dotenv


load_dotenv()

def load_config(config_path: str = None):
    """Load YAML config file.

    Behavior:
    - If `config_path` is provided, try to load it.
    - If not provided, try a small set of sensible defaults in order:
      1. ./config.yaml
      2. ./swingTrade_agent/config.yaml
      3. package-relative config (two levels up from this file)
    Raises FileNotFoundError with a helpful message if none are found.
    """

    possible_paths = []
    if config_path:
        possible_paths.append(config_path)

    # common repo root location
    possible_paths.append("config.yaml")
    # older default used in repo; keep for backwards compatibility
    possible_paths.append("swingTrade_agent/config.yaml")

    # package-relative: project_root/config.yaml
    package_root_cfg = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    possible_paths.append(package_root_cfg)

    for path in possible_paths:
        try:
            with open(path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        "config.yaml not found. Tried: {}. Provide a valid path to load_config(config_path=...)".format(
            ", ".join(possible_paths)
        )
    )

def get_env_var(key: str, default=None):
    """fetch environment variable  """

    return os.getenv(key, default)
