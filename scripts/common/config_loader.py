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


def load_config_defaults(config_file: str, backend: str = None) -> Dict[str, Any]:
    """
    Load default parameters from a configuration file.
    
    This function is useful when you want to load parameters from a YAML/JSON
    config file but get slide information from a CSV manifest. It returns the
    'defaults' section if present, otherwise returns all top-level parameters
    (excluding 'tasks' key and backend-specific sections).
    
    Backend-specific parameters (under 'slurm:', 'condor:', 'azure:' sections)
    are merged into the main config if the backend matches. If a 'cluster:' 
    subsection exists within the backend section, it is flattened into the 
    main parameters.
    
    Args:
        config_file: Path to configuration file (.json, .yaml, or .yml)
        backend: Backend name ('slurm', 'condor', or 'azure') to extract
                 backend-specific parameters. If None, backend sections are ignored.
        
    Returns:
        Dictionary containing default parameters with backend-specific params merged
    """
    config = load_config(config_file)
    
    # If there's a 'defaults' section, use it as base
    if 'defaults' in config:
        params = config['defaults'].copy()
    else:
        # Otherwise, return all parameters except 'tasks' and backend sections
        params = {k: v for k, v in config.items() 
                  if k not in ['tasks', 'slurm', 'condor', 'azure', 'azure_batch']}
    
    # If backend is specified, merge backend-specific parameters
    if backend:
        backend_key = backend.lower()
        # Also check for 'azure_batch' as an alias for 'azure'
        if backend_key == 'azure' and 'azure_batch' in config:
            backend_key = 'azure_batch'
        
        if backend_key in config and isinstance(config[backend_key], dict):
            backend_params = config[backend_key].copy()
            
            # If there's a 'cluster' subsection, flatten it into the main params
            if 'cluster' in backend_params and isinstance(backend_params['cluster'], dict):
                cluster_params = backend_params.pop('cluster')
                backend_params.update(cluster_params)
            
            # Merge backend-specific parameters (backend params override general params)
            params.update(backend_params)
    
    return params


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
