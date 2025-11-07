#!/usr/bin/env python3
"""
Configuration file loader for batch submission scripts.

Supports both JSON and YAML configuration files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set

try:
    import yaml
except ImportError:
    yaml = None


# Sensitive field names that should be excluded from manifests
SENSITIVE_FIELDS: Set[str] = {
    'aws_access_key_id',
    'aws_secret_access_key',
    'hf_token',
    'batch_account_key',
    'storage_account_key',
    'azure_files_share_name',
}


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_file: Path to configuration file (.json, .yaml, or .yml)
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    suffix = config_path.suffix.lower()
    
    with open(config_file, 'r') as f:
        if suffix == '.json':
            config = json.load(f)
        elif suffix in ['.yaml', '.yml']:
            if yaml is None:
                raise ImportError(
                    "PyYAML is required to load YAML configuration files. "
                    "Install it with: pip install PyYAML"
                )
            config = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {suffix}. "
                "Supported formats: .json, .yaml, .yml"
            )
    
    return config


def load_config_defaults(config_file: str) -> Dict[str, Any]:
    """
    Load default parameters from a configuration file.
    
    This function is useful when you want to load parameters from a YAML/JSON
    config file but get slide information from a CSV manifest. It returns the
    'defaults' section if present, otherwise returns all top-level parameters
    (excluding 'tasks' key).
    
    Args:
        config_file: Path to configuration file (.json, .yaml, or .yml)
        
    Returns:
        Dictionary containing default parameters
    """
    config = load_config(config_file)
    
    # If there's a 'defaults' section, return it
    if 'defaults' in config:
        return config['defaults']
    
    # Otherwise, return all parameters except 'tasks'
    return {k: v for k, v in config.items() if k != 'tasks'}


def filter_sensitive_fields(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove sensitive fields from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        New dictionary with sensitive fields removed
    """
    filtered = {}
    for key, value in config.items():
        if key not in SENSITIVE_FIELDS:
            if isinstance(value, dict):
                # Recursively filter nested dictionaries
                filtered[key] = filter_sensitive_fields(value)
            elif isinstance(value, list):
                # Filter lists of dictionaries
                filtered[key] = [
                    filter_sensitive_fields(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value
    return filtered


def add_config_to_metadata(
    task_metadata: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    task_id: str,
) -> None:
    """
    Add non-sensitive configuration to task metadata.
    
    Args:
        task_metadata: Dictionary mapping task_id to task metadata
        config: Configuration dictionary for the task
        task_id: Task identifier
    """
    if task_id not in task_metadata:
        task_metadata[task_id] = {}
    
    # Filter out sensitive fields before adding to metadata
    filtered_config = filter_sensitive_fields(config)
    
    # Add filtered config to metadata under 'config' key
    task_metadata[task_id]['config'] = filtered_config
