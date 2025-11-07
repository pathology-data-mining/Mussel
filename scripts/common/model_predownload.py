#!/usr/bin/env python3
"""
Model pre-download utility for batch processing.

This module handles automatic downloading and caching of foundation models
before batch job submission using the save_model CLI tool.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Optional, Dict


def can_model_be_saved(model_type: str) -> bool:
    """
    Check if a model type can be saved via save_model CLI.
    
    Models with empty paths in ModelType enum cannot be automatically saved
    and must be provided via configuration (e.g., CTRANSPATH, RESNET50).
    
    Args:
        model_type: Type of model (e.g., CTRANSPATH, CLIP, VIRCHOW)
        
    Returns:
        True if model can be saved, False otherwise
    """
    # Import here to avoid circular dependencies
    try:
        from mussel.models.model_factory import ModelType
        
        # Try to get the ModelType enum value
        try:
            model_enum = ModelType[model_type.upper()]
            # Check if path is None, empty, or whitespace
            return model_enum.path is not None and model_enum.path.strip() != ""
        except KeyError:
            # Unknown model type, assume it can't be saved
            return False
    except ImportError:
        # Can't import, assume it can be saved (fallback)
        return True


def run_save_model(model_type: str, output_path: str, model_path: Optional[str] = None) -> bool:
    """
    Run save_model CLI to download a model.
    
    Args:
        model_type: Type of model (e.g., CTRANSPATH, CLIP, VIRCHOW)
        output_path: Path to save the model
        model_path: Optional HuggingFace path for the model
        
    Returns:
        True if successful, False otherwise
    """
    # Build command arguments
    cmd_args = [f"model_type={model_type}", f"output_path={output_path}"]
    if model_path:
        cmd_args.append(f"model_path={model_path}")
    
    # Try to determine the best way to run save_model
    # Priority:
    # 1. Direct save_model if available in PATH
    # 2. uv run save_model if uv is available and works
    # 3. python -m mussel.cli.save_model as fallback
    
    commands_to_try = []
    
    # First, check if save_model is directly available
    if shutil.which('save_model'):
        commands_to_try.append(['save_model'] + cmd_args)
    
    # If uv is available, try uv run (but only if we're in a virtual environment)
    # IMPORTANT: Check both conditions to avoid SLURM issues:
    # - shutil.which('uv'): Ensures uv is available
    # - os.getenv('VIRTUAL_ENV'): Ensures we're in an activated venv
    # Without the VIRTUAL_ENV check, this would fail on SLURM submission nodes
    # where uv is available but the venv is not activated.
    if shutil.which('uv') and os.getenv('VIRTUAL_ENV'):
        commands_to_try.append(['uv', 'run', 'save_model'] + cmd_args)
    
    # Fallback: try python -m
    commands_to_try.append([sys.executable, '-m', 'mussel.cli.save_model'] + cmd_args)
    
    # Try each command in order
    last_error = None
    for cmd in commands_to_try:
        print(f"  Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  ✓ {model_type} saved to {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            last_error = e
            print(f"  ✗ Command failed: {e.stderr}", file=sys.stderr)
            # Try next command
            continue
        except FileNotFoundError as e:
            last_error = e
            # Try next command
            continue
    
    # All commands failed
    print(f"  ✗ Failed to download {model_type} after trying all methods", file=sys.stderr)
    if last_error:
        if isinstance(last_error, subprocess.CalledProcessError):
            print(f"  Last error: {last_error.stderr}", file=sys.stderr)
        else:
            print(f"  Last error: {last_error}", file=sys.stderr)
    return False


def pre_download_models(
    model_types: List[str],
    cache_dir: str,
    model_paths: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Pre-download all required models to a cache directory.
    
    Models without paths in ModelType enum (e.g., CTRANSPATH, RESNET50) cannot be
    automatically downloaded and will be skipped. These models must be provided
    via configuration using model_path parameters.
    
    Args:
        model_types: List of model types to download
        cache_dir: Directory to cache models
        model_paths: Optional dict of model_type -> HuggingFace path mappings
        
    Returns:
        Dict mapping model_type to cached file path (only for successfully downloaded models)
        
    Raises:
        RuntimeError: If any downloadable model fails to download
    """
    print(f"[Pre-download] Creating cache directory: {cache_dir}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    cached_models = {}
    model_paths = model_paths or {}
    skipped_models = []
    
    for model_type in model_types:
        # Check if model can be saved via save_model
        if not can_model_be_saved(model_type):
            print(f"[Pre-download] ⚠️  Skipping {model_type}: model does not have a default path in ModelType")
            print(f"[Pre-download]     {model_type} must be provided via configuration (e.g., prefilter_model_path)")
            skipped_models.append(model_type)
            continue
        
        # Generate cache filename
        cache_file = os.path.join(cache_dir, f"{model_type.lower()}.pth")
        
        # Skip if already cached
        if os.path.exists(cache_file):
            print(f"[Pre-download] {model_type} already cached: {cache_file}")
            cached_models[model_type] = cache_file
            continue
        
        print(f"[Pre-download] Downloading {model_type}...")
        hf_path = model_paths.get(model_type)
        
        if not run_save_model(model_type, cache_file, hf_path):
            raise RuntimeError(f"Failed to download model: {model_type}")
        
        cached_models[model_type] = cache_file
    
    if skipped_models:
        print(f"\n[Pre-download] ℹ️  Skipped models (must be provided via config): {', '.join(skipped_models)}")
    
    return cached_models


def upload_models_to_s3(cached_models: Dict[str, str], s3_prefix: str) -> Dict[str, str]:
    """
    Upload cached models to S3.
    
    Args:
        cached_models: Dict mapping model_type to local cache path
        s3_prefix: S3 prefix (e.g., s3://bucket/models/)
        
    Returns:
        Dict mapping model_type to S3 path
        
    Raises:
        RuntimeError: If S3 upload fails
    """
    try:
        import boto3
        from urllib.parse import urlparse
    except ImportError:
        raise RuntimeError("boto3 not installed. Install with: pip install boto3")
    
    s3_models = {}
    
    # Ensure s3_prefix ends with /
    if not s3_prefix.endswith('/'):
        s3_prefix += '/'
    
    # Parse S3 URL
    parsed = urlparse(s3_prefix)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    
    print(f"[Pre-download] Uploading models to S3: {s3_prefix}")
    
    try:
        # Create S3 client with optional custom endpoint
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        if endpoint_url:
            s3_client = boto3.client('s3', endpoint_url=endpoint_url)
            print(f"  Using custom S3 endpoint: {endpoint_url}")
        else:
            s3_client = boto3.client('s3')
        
        for model_type, local_path in cached_models.items():
            filename = os.path.basename(local_path)
            s3_key = f"{prefix}{filename}"
            s3_path = f"s3://{bucket}/{s3_key}"
            
            print(f"  Uploading {model_type}: {local_path} -> {s3_path}")
            s3_client.upload_file(local_path, bucket, s3_key)
            s3_models[model_type] = s3_path
            
        print(f"  ✓ Uploaded {len(s3_models)} models to S3")
        return s3_models
        
    except Exception as e:
        raise RuntimeError(f"Failed to upload models to S3: {e}")


if __name__ == "__main__":
    # Test
    models = ["CTRANSPATH", "CLIP"]
    cached = pre_download_models(models, "./test_cache")
    print("Cached models:", cached)
