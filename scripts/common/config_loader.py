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
    
    The 'resources:' section (if present) contains standardized resource parameters
    (cpus, memory, gpus) that are automatically mapped to backend-specific parameter
    names when a backend is specified.
    
    The 'aws:' section (if present) contains AWS configuration parameters that are
    mapped to aws_* parameter names (e.g., region -> aws_region, endpoint_url -> aws_endpoint_url).
    
    Args:
        config_file: Path to configuration file (.json, .yaml, or .yml)
        backend: Backend name ('slurm', 'condor', or 'azure') to map
                 standardized resource parameters to backend-specific names.
        
    Returns:
        Dictionary containing default parameters with backend-specific params merged
    """
    config = load_config(config_file)
    
    # If there's a 'defaults' section, use it as base
    if 'defaults' in config:
        params = config['defaults'].copy()
    else:
        # Otherwise, return all parameters except 'tasks', 'resources', 'aws', and backend sections
        params = {k: v for k, v in config.items() 
                  if k not in ['tasks', 'resources', 'aws', 'slurm', 'condor', 'azure', 'azure_batch']}
    
    # Process AWS configuration section and flatten with proper parameter names
    if 'aws' in config and isinstance(config['aws'], dict):
        aws_params = config['aws']
        # Map aws section parameters to top-level aws_* names
        if 'region' in aws_params:
            params['aws_region'] = aws_params['region']
        if 'endpoint_url' in aws_params:
            params['aws_endpoint_url'] = aws_params['endpoint_url']
        if 'access_key_id' in aws_params:
            params['aws_access_key_id'] = aws_params['access_key_id']
        if 'secret_access_key' in aws_params:
            params['aws_secret_access_key'] = aws_params['secret_access_key']
    
    # Process standardized resources section and map to backend-specific names
    if 'resources' in config and isinstance(config['resources'], dict):
        resources = config['resources']
        
        if backend:
            backend_key = backend.lower()
            
            # Map standardized resource names to backend-specific parameter names
            if backend_key == 'slurm':
                # Map: cpus -> cpus_per_task, memory -> mem, gpus -> gres
                if 'cpus' in resources:
                    params['cpus_per_task'] = resources['cpus']
                if 'memory' in resources:
                    params['mem'] = resources['memory']
                if 'gpus' in resources:
                    # For SLURM, gpus is specified in gres format
                    params['gres'] = f"gpu:{resources['gpus']}"
            elif backend_key == 'condor':
                # Map: cpus -> request_cpus, memory -> request_memory, gpus -> request_gpus
                if 'cpus' in resources:
                    params['request_cpus'] = resources['cpus']
                if 'memory' in resources:
                    params['request_memory'] = resources['memory']
                if 'gpus' in resources:
                    params['request_gpus'] = resources['gpus']
            else:
                # For other backends (azure), use as-is or map as needed
                params.update(resources)
        else:
            # No backend specified, include resources as-is
            params.update(resources)
    
    # If backend is specified, also merge backend-specific parameters
    if backend:
        backend_key = backend.lower()
        # Also check for 'azure_batch' as an alias for 'azure'
        if backend_key == 'azure' and 'azure_batch' in config:
            backend_key = 'azure_batch'
        
        if backend_key in config and isinstance(config[backend_key], dict):
            # Merge backend-specific parameters (backend params override general params and resources)
            params.update(config[backend_key])
    
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
