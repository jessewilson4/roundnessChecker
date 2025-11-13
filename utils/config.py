"""
# === UAIPCS START ===
file: config.py
purpose: Configuration management for image roundness analyzer - loads and validates settings from config.json
deps: [@json:library, @pathlib:library, @typing:library]
funcs:
  - load_config() -> dict  # side_effect: reads config.json file
  - save_config(config:dict) -> None  # side_effect: writes config.json file
  - get_setting(key_path:str, default:any) -> any  # no_side_effect
  - update_setting(key_path:str, value:any) -> None  # side_effect: updates config and saves file
refs:
  - config.json
notes: config_file=config.json, auto_create=true_if_missing, validation=basic_type_checking
# === UAIPCS END ===
"""

import json
from pathlib import Path
from typing import Any, Dict

CONFIG_FILE = Path(__file__).parent / 'config.json'

# Default configuration if file doesn't exist
DEFAULT_CONFIG = {
    "detection": {
        "confidence_threshold": 0.05,
        "closeup_threshold": 0.90,
        "comment": "confidence_threshold: minimum confidence for object detection (0.0-1.0). closeup_threshold: maximum bbox coverage to avoid closeups (0.0-1.0)"
    },
    "outliers": {
        "iqr_multiplier": 1.5,
        "method": "iqr",
        "comment": "iqr_multiplier: standard is 1.5 for IQR outlier detection. Lower = more strict (more outliers removed), Higher = more lenient. method: 'iqr' or 'zscore'"
    },
    "search": {
        "positive_keywords": ["isolated", "whole", "single", "white background"],
        "negative_keywords": ["group", "multiple", "crowd", "collage", "collection"],
        "comment": "positive_keywords: terms added to improve search quality. negative_keywords: terms to exclude from results"
    },
    "image_processing": {
        "max_resolution": 1024,
        "batch_size": 4,
        "thumbnail_size_kb": 25,
        "comment": "max_resolution: maximum dimension for image processing (px). batch_size: images processed per batch. thumbnail_size_kb: max thumbnail file size"
    },
    "api": {
        "images_per_search_default": 30,
        "images_dropdown_options": [10, 20, 30, 40, 50],
        "comment": "images_per_search_default: default number of images to fetch. images_dropdown_options: available options in dropdown"
    }
}

_config_cache = None


def load_config() -> Dict:
    """
    Load configuration from config.json.
    Creates default config if file doesn't exist.
    
    Returns:
        Configuration dictionary
    """
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    if not CONFIG_FILE.exists():
        print(f"⚠️  Config file not found, creating default: {CONFIG_FILE}")
        save_config(DEFAULT_CONFIG)
        _config_cache = DEFAULT_CONFIG.copy()
        return _config_cache
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            _config_cache = json.load(f)
        return _config_cache
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        print(f"   Using default configuration")
        _config_cache = DEFAULT_CONFIG.copy()
        return _config_cache


def save_config(config: Dict) -> None:
    """
    Save configuration to config.json.
    
    Args:
        config: Configuration dictionary to save
    """
    global _config_cache
    
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        _config_cache = config.copy()
        print(f"✓ Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"❌ Error saving config: {e}")


def get_setting(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration setting using dot notation.
    
    Args:
        key_path: Dot-separated path (e.g., 'detection.confidence_threshold')
        default: Default value if key doesn't exist
        
    Returns:
        Configuration value or default
        
    Examples:
        >>> get_setting('detection.confidence_threshold')
        0.05
        >>> get_setting('api.images_dropdown_options')
        [10, 20, 30, 40, 50]
    """
    config = load_config()
    keys = key_path.split('.')
    
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def update_setting(key_path: str, value: Any) -> None:
    """
    Update a configuration setting and save to file.
    
    Args:
        key_path: Dot-separated path (e.g., 'detection.confidence_threshold')
        value: New value to set
        
    Examples:
        >>> update_setting('detection.confidence_threshold', 0.10)
        >>> update_setting('api.images_dropdown_options', [10, 20, 30])
    """
    config = load_config()
    keys = key_path.split('.')
    
    # Navigate to the parent dict
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value
    
    # Save
    save_config(config)


def reload_config() -> Dict:
    """
    Force reload configuration from disk.
    
    Returns:
        Fresh configuration dictionary
    """
    global _config_cache
    _config_cache = None
    return load_config()


# Initialize config on import
load_config()