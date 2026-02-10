"""
Configuration management for the Cardiac Inpainting project.

This module provides utilities for loading, merging, and accessing
configuration from YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class Config:
    """
    Configuration class that supports nested attribute access.
    
    Example:
        config = Config.from_yaml("configs/default.yaml")
        print(config.training.batch_size)  # Access as attribute
        print(config["training"]["batch_size"])  # Access as dict
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
        """
        self._config = config_dict or {}
        
        # Convert nested dicts to Config objects for attribute access
        for key, value in self._config.items():
            if isinstance(value, dict):
                self._config[key] = Config(value)
    
    def __getattr__(self, name: str) -> Any:
        """Enable attribute-style access to config values."""
        if name.startswith('_'):
            return super().__getattribute__(name)
        
        if name in self._config:
            return self._config[name]
        
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to config values."""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config
    
    def __repr__(self) -> str:
        return f"Config({self._config})"
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value with a default fallback.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config back to a plain dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def keys(self):
        """Return configuration keys."""
        return self._config.keys()
    
    def items(self):
        """Return configuration items."""
        return self._config.items()
    
    def values(self):
        """Return configuration values."""
        return self._config.values()
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            Config object with loaded values
            
        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is invalid
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict or {})
    
    @classmethod
    def from_yaml_with_defaults(
        cls,
        yaml_path: Union[str, Path],
        defaults_path: Union[str, Path] = None
    ) -> "Config":
        """
        Load configuration from a YAML file, merging with defaults.
        
        Args:
            yaml_path: Path to the YAML configuration file
            defaults_path: Path to the defaults YAML file
            
        Returns:
            Config object with merged values
        """
        # Load defaults first
        if defaults_path:
            defaults = cls.from_yaml(defaults_path).to_dict()
        else:
            defaults = {}
        
        # Load main config
        main_config = cls.from_yaml(yaml_path).to_dict()
        
        # Merge configs (main overrides defaults)
        merged = deep_merge(defaults, main_config)
        
        return cls(merged)
    
    def save(self, yaml_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            yaml_path: Path to save the YAML configuration file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries. Values in override take precedence.
    
    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root
    """
    # Navigate up from src/config.py to project root
    current = Path(__file__).resolve()
    return current.parent.parent


def resolve_path(path: Union[str, Path], base_dir: Union[str, Path] = None) -> Path:
    """
    Resolve a path relative to a base directory.
    
    If the path is absolute, return it as-is.
    If relative, resolve it relative to the base_dir (or project root).
    
    Args:
        path: Path to resolve
        base_dir: Base directory for relative paths
        
    Returns:
        Resolved absolute path
    """
    path = Path(path)
    
    if path.is_absolute():
        return path
    
    if base_dir is None:
        base_dir = get_project_root()
    
    return Path(base_dir) / path


def load_config(
    config_path: Union[str, Path] = None,
    config_type: str = "default"
) -> Config:
    """
    Load a configuration file.
    
    Args:
        config_path: Explicit path to config file
        config_type: Type of config to load ("default", "training", "inference")
        
    Returns:
        Loaded configuration
    """
    if config_path is not None:
        return Config.from_yaml(config_path)
    
    # Load from standard location
    project_root = get_project_root()
    config_file = project_root / "configs" / f"{config_type}.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}. "
            f"Available types: default, training, inference"
        )
    
    # Always merge with defaults
    defaults_file = project_root / "configs" / "default.yaml"
    
    if config_type != "default" and defaults_file.exists():
        return Config.from_yaml_with_defaults(config_file, defaults_file)
    
    return Config.from_yaml(config_file)


# Convenience functions for common config access patterns
def get_device(config: Config) -> str:
    """Get the device string from config."""
    if hasattr(config, 'device') and config.device.get('cuda', True):
        device_id = config.device.get('device_id', 0)
        return f"cuda:{device_id}"
    return "cpu"


def get_paths(config: Config, base_dir: Path = None) -> Dict[str, Path]:
    """
    Get all paths from config as resolved absolute paths.
    
    Args:
        config: Configuration object
        base_dir: Base directory for resolving relative paths
        
    Returns:
        Dictionary of resolved paths
    """
    if base_dir is None:
        base_dir = get_project_root()
    
    paths = {}
    
    if hasattr(config, 'paths'):
        for key, value in config.paths.items():
            if isinstance(value, str):
                paths[key] = resolve_path(value, base_dir)
    
    return paths
