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
    # Detect uv environment
    cmd_prefix = []
    if shutil.which('uv'):
        if os.path.isdir('.venv') or os.getenv('VIRTUAL_ENV'):
            cmd_prefix = ['uv', 'run']
            print(f"  [Using uv environment]")
    
    cmd = cmd_prefix + ["save_model", f"model_type={model_type}", f"output_path={output_path}"]
    if model_path:
        cmd.append(f"model_path={model_path}")
    
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  ✓ {model_type} saved to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to download {model_type}: {e.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"  ✗ save_model command not found. Please ensure Mussel is installed.", file=sys.stderr)
        return False


def pre_download_models(
    model_types: List[str],
    cache_dir: str,
    model_paths: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Pre-download all required models to a cache directory.
    
    Args:
        model_types: List of model types to download
        cache_dir: Directory to cache models
        model_paths: Optional dict of model_type -> HuggingFace path mappings
        
    Returns:
        Dict mapping model_type to cached file path
        
    Raises:
        RuntimeError: If any model download fails
    """
    print(f"[Pre-download] Creating cache directory: {cache_dir}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    cached_models = {}
    model_paths = model_paths or {}
    
    for model_type in model_types:
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
