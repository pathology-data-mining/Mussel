#!/usr/bin/env python3
"""
Model pre-download utility for batch processing.

This module handles automatic downloading and caching of foundation models
before batch job submission.
"""

import os
import sys
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
    # Try direct import first
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
        # If mussel is not in the current environment, try using uv run
        import subprocess
        try:
            result = subprocess.run(
                ["uv", "run", "python", "-c", f"""
from mussel.models.model_factory import ModelType
try:
    model_enum = ModelType['{model_type.upper()}']
    print('1' if model_enum.path and model_enum.path.strip() else '0')
except KeyError:
    print('0')
"""],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip() == '1'
        except Exception:
            pass
        # Can't determine, assume it can be saved (fallback)
        return True


def run_save_model(model_type: str, output_path: str, model_path: Optional[str] = None) -> bool:
    """
    Download and save a model using the save_model CLI.
    
    Args:
        model_type: Type of model (e.g., CTRANSPATH, CLIP, VIRCHOW)
        output_path: Path to save the model
        model_path: Optional HuggingFace path for the model
        
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    
    print(f"  Downloading {model_type}...")
    
    try:
        # Pass through all relevant environment variables
        env = os.environ.copy()
        if 'HF_TOKEN' in env and 'HUGGINGFACE_TOKEN' not in env:
            env['HUGGINGFACE_TOKEN'] = env['HF_TOKEN']
        
        # Build command to call save_model CLI
        cmd = ["python", "-m", "mussel.cli.save_model",
               f"model_type={model_type}",
               f"output_path={output_path}"]
        
        if model_path:
            cmd.append(f"model_path={model_path}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        
        if result.returncode == 0:
            print(f"  ✓ {model_type} saved to {output_path}")
            return True
        else:
            print(f"  ✗ Failed to download {model_type}: {result.stderr}", file=sys.stderr)
            return False
            
    except Exception as e:
        print(f"  ✗ Failed to download {model_type}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
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
    
    # Filter models that can be downloaded
    downloadable_models = []
    for model_type in model_types:
        if not can_model_be_saved(model_type):
            print(f"[Pre-download] ⚠️  Skipping {model_type}: model does not have a default path in ModelType")
            print(f"[Pre-download]     {model_type} must be provided via configuration (e.g., prefilter_model_path)")
            skipped_models.append(model_type)
        else:
            downloadable_models.append(model_type)
    
    if not downloadable_models:
        print("[Pre-download] No downloadable models found")
        if skipped_models:
            print(f"[Pre-download] ℹ️  Skipped models (must be provided via config): {', '.join(skipped_models)}")
        return cached_models
    
    # Use batch download for efficiency
    print(f"[Pre-download] Downloading {len(downloadable_models)} models in batch...")
    if batch_download_models(downloadable_models, cache_dir):
        # Verify all models were downloaded
        for model_type in downloadable_models:
            cache_file_no_ext = os.path.join(cache_dir, model_type)
            cache_file_with_ext = os.path.join(cache_dir, f"{model_type.lower()}.pth")
            
            if os.path.exists(cache_file_no_ext):
                cached_models[model_type] = cache_file_no_ext
                print(f"[Pre-download] ✓ {model_type} cached at: {cache_file_no_ext}")
            elif os.path.exists(cache_file_with_ext):
                cached_models[model_type] = cache_file_with_ext
                print(f"[Pre-download] ✓ {model_type} cached at: {cache_file_with_ext}")
            else:
                raise RuntimeError(f"Model cache not found after download: {model_type}")
    else:
        raise RuntimeError("Batch download failed")
    
    if skipped_models:
        print(f"\n[Pre-download] ℹ️  Skipped models (must be provided via config): {', '.join(skipped_models)}")
    
    return cached_models


def batch_download_models(model_types: List[str], output_dir: str) -> bool:
    """
    Download multiple models in a single call using save_model CLI.
    
    Args:
        model_types: List of model types to download
        output_dir: Directory to save models
        
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    
    try:
        # Pass through all relevant environment variables
        env = os.environ.copy()
        if 'HF_TOKEN' in env and 'HUGGINGFACE_TOKEN' not in env:
            env['HUGGINGFACE_TOKEN'] = env['HF_TOKEN']
        
        # Build command with model_types list
        model_types_arg = f"[{','.join(model_types)}]"
        cmd = ["python", "-m", "mussel.cli.save_model",
               f"model_types={model_types_arg}",
               f"output_dir={output_dir}"]
        
        print(f"[Pre-download] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, timeout=3600)
        
        return result.returncode == 0
            
    except Exception as e:
        print(f"[Pre-download] ✗ Batch download failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False


def upload_models_to_s3(cached_models: Dict[str, str], s3_prefix: str) -> Dict[str, str]:
    """
    Upload cached models to S3 or Azure Blob Storage.
    
    Args:
        cached_models: Dict mapping model_type to local cache path (file or directory)
        s3_prefix: Storage prefix (e.g., s3://bucket/models/ or azblob://container/models/)
        
    Returns:
        Dict mapping model_type to storage path
        
    Raises:
        RuntimeError: If upload fails
    """
    from urllib.parse import urlparse
    
    # Parse storage URL
    parsed = urlparse(s3_prefix)
    scheme = parsed.scheme.lower()
    
    # Handle Azure Blob Storage
    if scheme == 'azblob':
        return _upload_models_to_azure_blob(cached_models, s3_prefix)
    
    # Handle S3
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 not installed. Install with: pip install boto3")
    
    s3_models = {}
    
    # Ensure s3_prefix ends with /
    if not s3_prefix.endswith('/'):
        s3_prefix += '/'
    
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
            # Check if path is a directory (for models like TITAN_SLIDE)
            if os.path.isdir(local_path):
                # Upload directory recursively
                dir_name = os.path.basename(local_path)
                s3_dir_prefix = f"{prefix}{dir_name}/"
                s3_path = f"s3://{bucket}/{s3_dir_prefix}"
                
                print(f"  Uploading {model_type} (directory): {local_path} -> {s3_path}")
                
                # Walk through directory and upload all files
                file_count = 0
                for root, dirs, files in os.walk(local_path):
                    for file in files:
                        local_file = os.path.join(root, file)
                        # Calculate relative path from base directory
                        rel_path = os.path.relpath(local_file, local_path)
                        s3_key = f"{s3_dir_prefix}{rel_path}"
                        
                        # Check if object already exists
                        try:
                            s3_client.head_object(Bucket=bucket, Key=s3_key)
                            print(f"    ✓ {model_type}/{rel_path} already exists, skipping")
                        except:
                            s3_client.upload_file(local_file, bucket, s3_key)
                            print(f"    ✓ Uploaded {model_type}/{rel_path}")
                        file_count += 1
                
                print(f"    ✓ Uploaded {file_count} files from {model_type}")
                s3_models[model_type] = s3_path
            else:
                # Upload single file
                filename = os.path.basename(local_path)
                s3_key = f"{prefix}{filename}"
                s3_path = f"s3://{bucket}/{s3_key}"
                
                # Check if object already exists
                try:
                    s3_client.head_object(Bucket=bucket, Key=s3_key)
                    print(f"  ✓ {model_type} already exists, skipping: {s3_path}")
                except:
                    print(f"  Uploading {model_type}: {local_path} -> {s3_path}")
                    s3_client.upload_file(local_path, bucket, s3_key)
                    print(f"  ✓ Uploaded {model_type}")
                
                s3_models[model_type] = s3_path
            
        print(f"  ✓ Uploaded {len(s3_models)} models to S3")
        return s3_models
        
    except Exception as e:
        raise RuntimeError(f"Failed to upload models to S3: {e}")


def _upload_models_to_azure_blob(cached_models: Dict[str, str], blob_prefix: str) -> Dict[str, str]:
    """
    Upload cached models to Azure Blob Storage.
    
    Args:
        cached_models: Dict mapping model_type to local cache path (file or directory)
        blob_prefix: Azure Blob prefix (e.g., azblob://container/models/)
        
    Returns:
        Dict mapping model_type to blob path
        
    Raises:
        RuntimeError: If upload fails
    """
    try:
        from azure.storage.blob import BlobServiceClient
        from urllib.parse import urlparse
    except ImportError:
        raise RuntimeError("azure-storage-blob not installed. Install with: pip install azure-storage-blob")
    
    blob_models = {}
    
    # Ensure blob_prefix ends with /
    if not blob_prefix.endswith('/'):
        blob_prefix += '/'
    
    # Parse Azure Blob URL: azblob://container/path/
    parsed = urlparse(blob_prefix)
    container_name = parsed.netloc
    prefix = parsed.path.lstrip('/')
    
    print(f"[Pre-download] Uploading models to Azure Blob: {blob_prefix}")
    
    try:
        # Get Azure Storage credentials from environment
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        account_name = os.environ.get("AZURE_STORAGE_ACCOUNT")
        account_key = os.environ.get("AZURE_STORAGE_KEY")
        
        if connection_string:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        elif account_name and account_key:
            account_url = f"https://{account_name}.blob.core.windows.net"
            blob_service_client = BlobServiceClient(account_url=account_url, credential=account_key)
        else:
            raise RuntimeError("Azure Storage credentials not found. Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_KEY")
        
        container_client = blob_service_client.get_container_client(container_name)
        
        for model_type, local_path in cached_models.items():
            # Check if path is a directory (for models like TITAN_SLIDE)
            if os.path.isdir(local_path):
                # Upload directory recursively
                dir_name = os.path.basename(local_path)
                blob_dir_prefix = f"{prefix}{dir_name}/"
                blob_path = f"azblob://{container_name}/{blob_dir_prefix}"
                
                print(f"  Uploading {model_type} (directory): {local_path} -> {blob_path}")
                
                # Walk through directory and upload all files
                file_count = 0
                for root, dirs, files in os.walk(local_path):
                    for file in files:
                        local_file = os.path.join(root, file)
                        # Calculate relative path from base directory
                        rel_path = os.path.relpath(local_file, local_path)
                        blob_name = f"{blob_dir_prefix}{rel_path}"
                        
                        blob_client = container_client.get_blob_client(blob_name)
                        
                        # Check if blob already exists
                        if blob_client.exists():
                            print(f"    ✓ {model_type}/{rel_path} already exists, skipping")
                        else:
                            with open(local_file, "rb") as data:
                                blob_client.upload_blob(data, overwrite=False)
                            print(f"    ✓ Uploaded {model_type}/{rel_path}")
                        file_count += 1
                
                print(f"    ✓ Uploaded {file_count} files from {model_type}")
                blob_models[model_type] = blob_path
            else:
                # Upload single file
                filename = os.path.basename(local_path)
                blob_name = f"{prefix}{filename}"
                blob_path = f"azblob://{container_name}/{blob_name}"
                
                blob_client = container_client.get_blob_client(blob_name)
                
                # Check if blob already exists
                if blob_client.exists():
                    print(f"  ✓ {model_type} already exists, skipping: {blob_path}")
                else:
                    print(f"  Uploading {model_type}: {local_path} -> {blob_path}")
                    with open(local_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=False)
                    print(f"  ✓ Uploaded {model_type}")
                
                blob_models[model_type] = blob_path
            
        print(f"  ✓ Uploaded {len(blob_models)} models to Azure Blob")
        return blob_models
        
    except Exception as e:
        raise RuntimeError(f"Failed to upload models to Azure Blob: {e}")


if __name__ == "__main__":
    # Test
    models = ["CTRANSPATH", "CLIP"]
    cached = pre_download_models(models, "./test_cache")
    print("Cached models:", cached)
