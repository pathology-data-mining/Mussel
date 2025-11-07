#!/usr/bin/env python3
"""
Azure Files staging module for preprocessing slides and models.

This module handles:
- Uploading slides from local/S3 to Azure Files
- Uploading models to Azure Files
- Cleaning up staged files after processing

Azure Files can be mounted to Azure Batch containers for direct access,
eliminating download overhead during task execution.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

try:
    from azure.storage.fileshare import ShareServiceClient, ShareFileClient
    from azure.core.exceptions import ResourceNotFoundError
except ImportError:
    print("WARNING: Azure Files SDK not installed.")
    print("Install with: pip install azure-storage-file-share")
    ShareServiceClient = None
    ShareFileClient = None
    ResourceNotFoundError = None


class AzureFilesStaging:
    """Handles staging files to Azure Files for Azure Batch processing."""
    
    def __init__(
        self,
        account_name: str,
        account_key: str,
        share_name: str = "mussel-staging",
    ):
        """
        Initialize Azure Files staging client.
        
        Args:
            account_name: Azure Storage account name
            account_key: Azure Storage account key
            share_name: File share name (created if doesn't exist)
        """
        if ShareServiceClient is None:
            raise ImportError("azure-storage-file-share package required")
        
        self.account_name = account_name
        self.account_key = account_key
        self.share_name = share_name
        
        # Initialize Azure Files client
        account_url = f"https://{account_name}.file.core.windows.net"
        self.service_client = ShareServiceClient(
            account_url=account_url,
            credential=account_key,
        )
        
        # Ensure share exists
        self._ensure_share_exists()
        
    def _ensure_share_exists(self) -> None:
        """Create file share if it doesn't exist."""
        try:
            share_client = self.service_client.get_share_client(self.share_name)
            share_client.get_share_properties()
            print(f"[Azure Files] Using existing share: {self.share_name}")
        except ResourceNotFoundError:
            print(f"[Azure Files] Creating share: {self.share_name}")
            self.service_client.create_share(self.share_name)
    
    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        show_progress: bool = True,
    ) -> str:
        """
        Upload a single file to Azure Files.
        
        Args:
            local_path: Local file path
            remote_path: Remote path within share (e.g., "slides/slide_001.svs")
            show_progress: Show upload progress
            
        Returns:
            Remote path in Azure Files
        """
        # Download from S3 if needed
        if local_path.startswith("s3://"):
            print(f"[Azure Files] Staging from S3: {local_path}")
            temp_path = self._download_from_s3(local_path)
            local_path = temp_path
            cleanup_temp = True
        else:
            cleanup_temp = False
        
        # Create parent directory in share
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            self._create_directory(remote_dir)
        
        # Upload file
        print(f"[Azure Files] Uploading: {remote_path}")
        share_client = self.service_client.get_share_client(self.share_name)
        file_client = share_client.get_file_client(remote_path)
        
        with open(local_path, "rb") as source_file:
            file_client.upload_file(source_file)
        
        # Cleanup temp file if downloaded from S3
        if cleanup_temp:
            os.remove(local_path)
        
        print(f"[Azure Files] Uploaded: {remote_path}")
        return remote_path
    
    def upload_directory(
        self,
        local_dir: str,
        remote_dir: str,
    ) -> List[str]:
        """
        Upload all files in a directory to Azure Files.
        
        Args:
            local_dir: Local directory path
            remote_dir: Remote directory within share
            
        Returns:
            List of uploaded remote paths
        """
        uploaded_files = []
        
        for root, dirs, files in os.walk(local_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                rel_path = os.path.relpath(local_path, local_dir)
                remote_path = os.path.join(remote_dir, rel_path).replace("\\", "/")
                
                self.upload_file(local_path, remote_path, show_progress=False)
                uploaded_files.append(remote_path)
        
        return uploaded_files
    
    def _create_directory(self, directory: str) -> None:
        """Create directory structure in file share."""
        share_client = self.service_client.get_share_client(self.share_name)
        
        # Create each parent directory
        parts = directory.split("/")
        current_path = ""
        
        for part in parts:
            if not part:
                continue
            
            current_path = f"{current_path}/{part}" if current_path else part
            
            try:
                dir_client = share_client.get_directory_client(current_path)
                dir_client.get_directory_properties()
            except ResourceNotFoundError:
                dir_client.create_directory()
    
    def _download_from_s3(self, s3_path: str) -> str:
        """Download file from S3 to temporary location."""
        import tempfile
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(s3_path)[1])
        temp_path = temp_file.name
        temp_file.close()
        
        # Download from S3 using AWS CLI
        # Using list arguments for subprocess.run is safe against command injection
        cmd = ["aws", "s3", "cp", s3_path, temp_path]
        
        # Add custom endpoint URL if specified
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        if endpoint_url:
            # Insert endpoint URL as separate list elements for safety
            cmd.insert(2, "--endpoint-url")
            cmd.insert(3, endpoint_url)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download from S3: {result.stderr}")
        
        return temp_path
    
    def delete_file(self, remote_path: str) -> None:
        """Delete a file from Azure Files."""
        try:
            share_client = self.service_client.get_share_client(self.share_name)
            file_client = share_client.get_file_client(remote_path)
            file_client.delete_file()
            print(f"[Azure Files] Deleted: {remote_path}")
        except ResourceNotFoundError:
            pass  # File already deleted
    
    def delete_directory(self, directory: str) -> None:
        """Delete a directory and all its contents from Azure Files."""
        try:
            share_client = self.service_client.get_share_client(self.share_name)
            dir_client = share_client.get_directory_client(directory)
            
            # List and delete all files
            for item in dir_client.list_directories_and_files():
                item_path = f"{directory}/{item['name']}"
                
                if item['is_directory']:
                    self.delete_directory(item_path)
                else:
                    self.delete_file(item_path)
            
            # Delete the directory itself
            dir_client.delete_directory()
            print(f"[Azure Files] Deleted directory: {directory}")
        except ResourceNotFoundError:
            pass  # Directory already deleted
    
    def cleanup_staging(self, directories: List[str]) -> None:
        """
        Clean up staged files and directories.
        
        Args:
            directories: List of directory paths to delete
        """
        print(f"[Azure Files] Cleaning up {len(directories)} directories...")
        for directory in directories:
            self.delete_directory(directory)
        print("[Azure Files] Cleanup complete")
    
    def get_mount_config(self) -> Dict:
        """
        Get Azure Files mount configuration for Azure Batch.
        
        Returns:
            Mount configuration dict for pool creation
        """
        return {
            "azure_file_share_name": self.share_name,
            "account_name": self.account_name,
            "account_key": self.account_key,
            "relative_mount_path": "azfiles",
            "mount_options": "-o vers=3.0,dir_mode=0777,file_mode=0777,sec=ntlmssp",
        }


def stage_slides_to_azure_files(
    slides: List[Dict[str, str]],
    account_name: str,
    account_key: str,
    share_name: str = "mussel-staging",
    remote_dir: str = "slides",
) -> tuple:
    """
    Stage slides to Azure Files before processing.
    
    Args:
        slides: List of dicts with 'slide_id' and 'slide_path' keys
        account_name: Azure Storage account name
        account_key: Azure Storage account key
        share_name: File share name
        remote_dir: Remote directory for slides
        
    Returns:
        (staging_client, staged_paths) tuple
    """
    staging = AzureFilesStaging(account_name, account_key, share_name)
    
    staged_paths = {}
    print(f"[Azure Files] Staging {len(slides)} slides...")
    
    for slide in slides:
        slide_id = slide['slide_id']
        slide_path = slide['slide_path']
        
        # Determine filename
        if slide_path.startswith("s3://") or slide_path.startswith("http"):
            filename = os.path.basename(slide_path)
        else:
            filename = os.path.basename(slide_path)
        
        # Upload to Azure Files
        remote_path = f"{remote_dir}/{filename}"
        staging.upload_file(slide_path, remote_path)
        
        # Store mapping
        staged_paths[slide_id] = remote_path
    
    print(f"[Azure Files] Staged {len(staged_paths)} slides to share '{share_name}'")
    return staging, staged_paths


def stage_models_to_azure_files(
    model_paths: Dict[str, str],
    account_name: str,
    account_key: str,
    share_name: str = "mussel-staging",
    remote_dir: str = "models",
) -> Dict[str, str]:
    """
    Stage model files to Azure Files.
    
    Args:
        model_paths: Dict mapping model types to local paths
        account_name: Azure Storage account name
        account_key: Azure Storage account key
        share_name: File share name
        remote_dir: Remote directory for models
        
    Returns:
        Dict mapping model types to remote paths
    """
    staging = AzureFilesStaging(account_name, account_key, share_name)
    
    staged_models = {}
    print(f"[Azure Files] Staging {len(model_paths)} models...")
    
    for model_type, local_path in model_paths.items():
        if not local_path:
            continue
        
        filename = os.path.basename(local_path)
        remote_path = f"{remote_dir}/{filename}"
        
        staging.upload_file(local_path, remote_path)
        staged_models[model_type] = remote_path
    
    print(f"[Azure Files] Staged {len(staged_models)} models")
    return staged_models
