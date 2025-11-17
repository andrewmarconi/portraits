"""Configuration management for Portraits."""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Centralized configuration manager.

    Loads configuration from config.yaml and provides convenient access
    to settings with environment variable expansion and defaults.

    Examples:
        >>> config = Config()
        >>> model_id = config.get('models.sdxl_turbo')
        >>> output_dir = config.get('paths.output_dir')
    """

    def __init__(self, config_path: str | None = None):
        """Initialize configuration.

        Args:
            config_path: Path to config file (default: ./config.yaml)
        """
        if config_path is None:
            # Look for config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "portraits" / "config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Please create config.yaml in the project root."
            )

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        return self._expand_env_vars(config)

    def _expand_env_vars(self, obj: Any) -> Any:
        """Recursively expand environment variables in config values.

        Supports syntax: ${VAR_NAME:default_value}
        """
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Expand environment variables
            if obj.startswith("${") and "}" in obj:
                var_expr = obj[2 : obj.index("}")]
                if ":" in var_expr:
                    var_name, default = var_expr.split(":", 1)
                    return os.environ.get(var_name, default)
                else:
                    return os.environ.get(var_expr, obj)
            return obj
        else:
            return obj

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., 'models.sdxl_turbo')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config.get('models.sdxl_turbo')
            'stabilityai/sdxl-turbo'
            >>> config.get('image_generation.width')
            512
            >>> config.get('nonexistent.key', 'fallback')
            'fallback'
        """
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value (runtime only, not persisted).

        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split(".")
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()


# Global config instance (singleton pattern)
_config_instance: Config | None = None


def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Config: Global configuration object
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


# Convenience alias
config = get_config()
