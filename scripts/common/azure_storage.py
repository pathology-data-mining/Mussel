#!/usr/bin/env python3
"""
Azure Storage utilities for distributed processing.

Provides functions for:
- Azure Blob Storage: slide staging and result publishing
- Azure Files: slide upload/download and model storage
- Mixed storage support (Azure + S3)
"""

import os
import sys
import subprocess
from typing import Optional, Tuple
from pathlib import Path


def parse_azure_path(path: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse Azure storage path.
    
    Supported formats:
    - Azure Blob: azblob://account/container/path
    - Azure Files: azfiles://account/share/path
    
    Returns:
        Tuple of (storage_type, account_name, container_or_share, blob_or_file_path)
        or None if not an Azure path
    """
    if not path.startswith(('azblob://', 'azfiles://')):
        return None
    
    if path.startswith('azblob://'):
        storage_type = 'blob'
        path = path[9:]  # Remove 'azblob://'
    else:
        storage_type = 'files'
        path = path[10:]  # Remove 'azfiles://'
    
    parts = path.split('/', 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid Azure path format. Expected: az{storage_type}://account/container/path")
    
    account_name, container_or_share, blob_path = parts
    return storage_type, account_name, container_or_share, blob_path


def download_from_azure_blob(blob_url: str, local_path: str, 
                              account_name: Optional[str] = None,
                              account_key: Optional[str] = None) -> bool:
    """
    Download file from Azure Blob Storage using Azure CLI.
    
    Args:
        blob_url: Azure Blob URL (azblob://account/container/path)
        local_path: Local destination path
        account_name: Azure storage account name (optional if in URL)
        account_key: Azure storage account key (or use env: AZURE_STORAGE_KEY)
    
    Returns:
        True if successful, False otherwise
    """
    parsed = parse_azure_path(blob_url)
    if not parsed or parsed[0] != 'blob':
        print(f"Error: Invalid Azure Blob URL: {blob_url}")
        return False
    
    _, acc_name, container, blob_path = parsed
    account_name = account_name or acc_name
    
    # Create local directory if needed
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Build az storage blob download command
    cmd = [
        'az', 'storage', 'blob', 'download',
        '--account-name', account_name,
        '--container-name', container,
        '--name', blob_path,
        '--file', local_path,
    ]
    
    # Add account key if provided
    if account_key:
        cmd.extend(['--account-key', account_key])
    elif os.getenv('AZURE_STORAGE_KEY'):
        cmd.extend(['--account-key', os.getenv('AZURE_STORAGE_KEY')])
    
    print(f"[Azure Blob] Downloading: {blob_url} -> {local_path}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[Azure Blob] Download successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Azure Blob] Download failed: {e.stderr}")
        return False


def upload_to_azure_blob(local_path: str, blob_url: str,
                         account_name: Optional[str] = None,
                         account_key: Optional[str] = None) -> bool:
    """
    Upload file to Azure Blob Storage using Azure CLI.
    
    Args:
        local_path: Local file path
        blob_url: Azure Blob URL (azblob://account/container/path)
        account_name: Azure storage account name (optional if in URL)
        account_key: Azure storage account key (or use env: AZURE_STORAGE_KEY)
    
    Returns:
        True if successful, False otherwise
    """
    parsed = parse_azure_path(blob_url)
    if not parsed or parsed[0] != 'blob':
        print(f"Error: Invalid Azure Blob URL: {blob_url}")
        return False
    
    _, acc_name, container, blob_path = parsed
    account_name = account_name or acc_name
    
    if not os.path.exists(local_path):
        print(f"Error: Local file not found: {local_path}")
        return False
    
    # Build az storage blob upload command
    cmd = [
        'az', 'storage', 'blob', 'upload',
        '--account-name', account_name,
        '--container-name', container,
        '--name', blob_path,
        '--file', local_path,
        '--overwrite',
    ]
    
    # Add account key if provided
    if account_key:
        cmd.extend(['--account-key', account_key])
    elif os.getenv('AZURE_STORAGE_KEY'):
        cmd.extend(['--account-key', os.getenv('AZURE_STORAGE_KEY')])
    
    print(f"[Azure Blob] Uploading: {local_path} -> {blob_url}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[Azure Blob] Upload successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Azure Blob] Upload failed: {e.stderr}")
        return False


def download_from_azure_files(file_url: str, local_path: str,
                               account_name: Optional[str] = None,
                               account_key: Optional[str] = None) -> bool:
    """
    Download file from Azure Files using Azure CLI.
    
    Args:
        file_url: Azure Files URL (azfiles://account/share/path)
        local_path: Local destination path
        account_name: Azure storage account name (optional if in URL)
        account_key: Azure storage account key (or use env: AZURE_STORAGE_KEY)
    
    Returns:
        True if successful, False otherwise
    """
    parsed = parse_azure_path(file_url)
    if not parsed or parsed[0] != 'files':
        print(f"Error: Invalid Azure Files URL: {file_url}")
        return False
    
    _, acc_name, share, file_path = parsed
    account_name = account_name or acc_name
    
    # Create local directory if needed
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Build az storage file download command
    cmd = [
        'az', 'storage', 'file', 'download',
        '--account-name', account_name,
        '--share-name', share,
        '--path', file_path,
        '--dest', local_path,
    ]
    
    # Add account key if provided
    if account_key:
        cmd.extend(['--account-key', account_key])
    elif os.getenv('AZURE_STORAGE_KEY'):
        cmd.extend(['--account-key', os.getenv('AZURE_STORAGE_KEY')])
    
    print(f"[Azure Files] Downloading: {file_url} -> {local_path}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[Azure Files] Download successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Azure Files] Download failed: {e.stderr}")
        return False


def upload_to_azure_files(local_path: str, file_url: str,
                          account_name: Optional[str] = None,
                          account_key: Optional[str] = None) -> bool:
    """
    Upload file to Azure Files using Azure CLI.
    
    Args:
        local_path: Local file path
        file_url: Azure Files URL (azfiles://account/share/path)
        account_name: Azure storage account name (optional if in URL)
        account_key: Azure storage account key (or use env: AZURE_STORAGE_KEY)
    
    Returns:
        True if successful, False otherwise
    """
    parsed = parse_azure_path(file_url)
    if not parsed or parsed[0] != 'files':
        print(f"Error: Invalid Azure Files URL: {file_url}")
        return False
    
    _, acc_name, share, file_path = parsed
    account_name = account_name or acc_name
    
    if not os.path.exists(local_path):
        print(f"Error: Local file not found: {local_path}")
        return False
    
    # Create parent directories in Azure Files if needed
    file_dir = os.path.dirname(file_path)
    if file_dir:
        mkdir_cmd = [
            'az', 'storage', 'directory', 'create',
            '--account-name', account_name,
            '--share-name', share,
            '--name', file_dir,
        ]
        if account_key:
            mkdir_cmd.extend(['--account-key', account_key])
        elif os.getenv('AZURE_STORAGE_KEY'):
            mkdir_cmd.extend(['--account-key', os.getenv('AZURE_STORAGE_KEY')])
        
        # Ignore errors if directory already exists
        subprocess.run(mkdir_cmd, capture_output=True)
    
    # Build az storage file upload command
    cmd = [
        'az', 'storage', 'file', 'upload',
        '--account-name', account_name,
        '--share-name', share,
        '--source', local_path,
        '--path', file_path,
    ]
    
    # Add account key if provided
    if account_key:
        cmd.extend(['--account-key', account_key])
    elif os.getenv('AZURE_STORAGE_KEY'):
        cmd.extend(['--account-key', os.getenv('AZURE_STORAGE_KEY')])
    
    print(f"[Azure Files] Uploading: {local_path} -> {file_url}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[Azure Files] Upload successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Azure Files] Upload failed: {e.stderr}")
        return False


def is_azure_path(path: str) -> bool:
    """Check if path is an Azure storage path."""
    return path.startswith(('azblob://', 'azfiles://'))


def download_from_azure(azure_url: str, local_path: str,
                        account_name: Optional[str] = None,
                        account_key: Optional[str] = None) -> bool:
    """
    Download from Azure storage (auto-detect Blob or Files).
    
    Args:
        azure_url: Azure URL (azblob:// or azfiles://)
        local_path: Local destination path
        account_name: Azure storage account name
        account_key: Azure storage account key
    
    Returns:
        True if successful, False otherwise
    """
    if azure_url.startswith('azblob://'):
        return download_from_azure_blob(azure_url, local_path, account_name, account_key)
    elif azure_url.startswith('azfiles://'):
        return download_from_azure_files(azure_url, local_path, account_name, account_key)
    else:
        print(f"Error: Not an Azure URL: {azure_url}")
        return False


def upload_to_azure(local_path: str, azure_url: str,
                   account_name: Optional[str] = None,
                   account_key: Optional[str] = None) -> bool:
    """
    Upload to Azure storage (auto-detect Blob or Files).
    
    Args:
        local_path: Local file path
        azure_url: Azure URL (azblob:// or azfiles://)
        account_name: Azure storage account name
        account_key: Azure storage account key
    
    Returns:
        True if successful, False otherwise
    """
    if azure_url.startswith('azblob://'):
        return upload_to_azure_blob(local_path, azure_url, account_name, account_key)
    elif azure_url.startswith('azfiles://'):
        return upload_to_azure_files(local_path, azure_url, account_name, account_key)
    else:
        print(f"Error: Not an Azure URL: {azure_url}")
        return False


if __name__ == '__main__':
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Azure storage utility')
    parser.add_argument('action', choices=['download', 'upload'])
    parser.add_argument('source', help='Source path')
    parser.add_argument('dest', help='Destination path')
    parser.add_argument('--account-name', help='Azure storage account name')
    parser.add_argument('--account-key', help='Azure storage account key')
    
    args = parser.parse_args()
    
    if args.action == 'download':
        success = download_from_azure(args.source, args.dest, args.account_name, args.account_key)
    else:
        success = upload_to_azure(args.source, args.dest, args.account_name, args.account_key)
    
    sys.exit(0 if success else 1)
