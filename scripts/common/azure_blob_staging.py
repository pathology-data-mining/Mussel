#!/usr/bin/env python3
"""
Azure Blob Storage staging module for preprocessing slides and models.

This module handles:
- Uploading slides from local/S3 to Azure Blob Storage
- Uploading models to Azure Blob Storage
- Cleaning up staged files after processing

Azure Blob Storage provides better scalability and performance compared to Azure Files
for large-scale batch processing workloads.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

try:
    from azure.storage.blob import BlobServiceClient, BlobClient
    from azure.core.exceptions import ResourceNotFoundError
except ImportError:
    print("WARNING: Azure Blob SDK not installed.")
    print("Install with: pip install azure-storage-blob")
    BlobServiceClient = None
    BlobClient = None
    ResourceNotFoundError = None


class AzureBlobStaging:
    """Handles staging files to Azure Blob Storage for Azure Batch processing."""
    
    def __init__(
        self,
        account_name: str,
        account_key: str,
        container_name: str = "mussel-staging",
    ):
        """
        Initialize Azure Blob Storage staging client.
        
        Args:
            account_name: Azure Storage account name
            account_key: Azure Storage account key
            container_name: Blob container name (created if doesn't exist)
        """
        if BlobServiceClient is None:
            raise ImportError("azure-storage-blob package required")
        
        self.account_name = account_name
        self.account_key = account_key
        self.container_name = container_name
        
        # Initialize Azure Blob client
        account_url = f"https://{account_name}.blob.core.windows.net"
        self.service_client = BlobServiceClient(
            account_url=account_url,
            credential=account_key,
        )
        
        # Ensure container exists
        self._ensure_container_exists()
        
    def _ensure_container_exists(self) -> None:
        """Create blob container if it doesn't exist."""
        try:
            container_client = self.service_client.get_container_client(self.container_name)
            container_client.get_container_properties()
            print(f"[Azure Blob] Using existing container: {self.container_name}")
        except ResourceNotFoundError:
            print(f"[Azure Blob] Creating container: {self.container_name}")
            self.service_client.create_container(self.container_name)
    
    def blob_exists(self, blob_name: str) -> bool:
        """
        Check if a blob exists in Azure Blob Storage.
        
        Args:
            blob_name: Blob name within container
            
        Returns:
            True if blob exists, False otherwise
        """
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            blob_client.get_blob_properties()
            return True
        except:
            return False
    
    def upload_file(
        self,
        local_path: str,
        blob_name: str,
        show_progress: bool = True,
        skip_if_exists: bool = True,
    ) -> str:
        """
        Upload a file to Azure Blob Storage.
        
        Handles three scenarios:
        1. Local file path: upload directly
        2. S3 path (s3://...): download to temp, then upload
        3. Already an azblob:// path: verify and return
        
        Args:
            local_path: Local file path or S3 URL
            blob_name: Destination blob name in container
            show_progress: Show upload progress (not implemented yet)
            skip_if_exists: Skip upload if blob already exists
            
        Returns:
            azblob:// URL for the uploaded blob
        """
        # If already an azblob path, just return it
        if local_path.startswith("azblob://"):
            return local_path
        
        # Check if blob already exists
        if skip_if_exists and self.blob_exists(blob_name):
            if show_progress:
                print(f"    [SKIP] Blob already exists: {blob_name}")
            return f"azblob://{self.account_name}/{self.container_name}/{blob_name}"
        
        # Handle S3 download
        temp_file = None
        # Default to using azcopy if available (much faster than SDK)
        use_azcopy = os.environ.get("USE_AZCOPY", "true").lower() == "true"
        
        if local_path.startswith("s3://"):
            if show_progress:
                print(f"    [S3] Downloading from S3: {local_path}")
            
            # Download from S3 to temp file
            temp_dir = Path(os.path.expanduser("~/tmp/azure_blob_staging"))
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            filename = os.path.basename(blob_name)
            temp_file = temp_dir / filename
            
            # Use AWS CLI for download
            cmd = ["aws", "s3", "cp", local_path, str(temp_file)]
            
            # Add endpoint URL if set (for MinIO compatibility)
            endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
            if endpoint_url:
                cmd.extend(["--endpoint-url", endpoint_url])
            
            # Debug: Log the command and environment
            if show_progress:
                print(f"    [DEBUG] Running command: {' '.join(cmd)}")
                print(f"    [DEBUG] AWS_ACCESS_KEY_ID: {os.environ.get('AWS_ACCESS_KEY_ID')}")
                print(f"    [DEBUG] AWS_SECRET_ACCESS_KEY: {'***' if os.environ.get('AWS_SECRET_ACCESS_KEY') else 'NOT SET'}")
                print(f"    [DEBUG] AWS_ENDPOINT_URL: {endpoint_url}")
            
            try:
                # Explicitly pass environment to ensure AWS credentials are available
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=os.environ.copy())
                local_path = str(temp_file)
            except subprocess.CalledProcessError as e:
                # Include full error details for debugging
                error_msg = f"Failed to download from S3: {e.stderr}"
                if show_progress:
                    print(f"    [DEBUG] Error details: {error_msg}")
                raise RuntimeError(error_msg)
        
        # Verify local file exists
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")
        
        # Upload to Azure Blob Storage
        try:
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            
            if show_progress:
                print(f"    [UPLOAD] Uploading {file_size_mb:.1f} MB to Azure Blob...")
            
            # Try using azcopy for better performance if available
            if use_azcopy and shutil.which("azcopy"):
                if show_progress:
                    print(f"    [AZCOPY] Using azcopy for faster upload")
                else:
                    # Always log when using azcopy, even if show_progress=False
                    print(f"    [INFO] Using azcopy for {os.path.basename(local_path)} ({file_size_mb:.1f} MB)")
                
                dest_url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
                
                # Set AZCOPY_AUTO_LOGIN_TYPE to prevent interactive prompts
                env = os.environ.copy()
                env["AZCOPY_AUTO_LOGIN_TYPE"] = "SPN"  # Service Principal
                env["AZCOPY_SPA_APPLICATION_ID"] = ""  # Not used with account key
                env["AZCOPY_SPA_CLIENT_SECRET"] = self.account_key
                
                # azcopy uses account key via URL with SAS token or env var
                # For account key auth, we need to generate a SAS token or use env var
                cmd = [
                    "azcopy", "copy",
                    local_path,
                    dest_url,
                    "--overwrite=true",
                    "--blob-type=BlockBlob"
                ]
                
                # Add account key via environment variable
                env["AZCOPY_ACCOUNT_KEY"] = self.account_key
                
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
                    if show_progress:
                        print(f"    [DONE] azcopy upload complete")
                except subprocess.CalledProcessError as e:
                    if show_progress:
                        print(f"    [WARN] azcopy failed, falling back to SDK: {e.stderr}")
                    # Fall back to SDK upload
                    use_azcopy = False
            
            # Fall back to SDK upload if azcopy not used or failed
            if not use_azcopy or not shutil.which("azcopy"):
                if not shutil.which("azcopy") and use_azcopy:
                    print(f"    [INFO] azcopy not found in PATH, using Python SDK")
                
                blob_client = self.service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob_name
                )
                
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                
                if show_progress:
                    print(f"    [DONE] Upload complete")
            
        finally:
            # Clean up temp file if we downloaded from S3
            if temp_file and temp_file.exists():
                temp_file.unlink()
        
        return f"azblob://{self.account_name}/{self.container_name}/{blob_name}"
    
    def upload_directory(
        self,
        local_dir: str,
        blob_prefix: str,
        show_progress: bool = True,
    ) -> str:
        """
        Upload a directory to Azure Blob Storage.
        
        Args:
            local_dir: Local directory path
            blob_prefix: Blob prefix (directory path in container)
            show_progress: Show upload progress
            
        Returns:
            Azure Blob URL prefix for the directory
        """
        local_dir_path = Path(local_dir)
        if not local_dir_path.is_dir():
            raise ValueError(f"Not a directory: {local_dir}")
        
        if show_progress:
            print(f"    [UPLOAD DIR] Uploading directory to Azure Blob...")
        
        # Upload all files in the directory
        for file_path in local_dir_path.rglob("*"):
            if file_path.is_file():
                # Calculate relative path from local_dir
                rel_path = file_path.relative_to(local_dir_path)
                blob_name = f"{blob_prefix}/{rel_path}".replace("\\", "/")
                
                blob_client = self.service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob_name
                )
                
                # Check if blob already exists
                try:
                    blob_properties = blob_client.get_blob_properties()
                    if show_progress:
                        file_size_kb = file_path.stat().st_size / 1024
                        print(f"      Skipped {rel_path} (already exists, {file_size_kb:.1f} KB)")
                    continue
                except Exception:
                    pass
                
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=False)
                
                if show_progress:
                    file_size_kb = file_path.stat().st_size / 1024
                    print(f"      Uploaded {rel_path} ({file_size_kb:.1f} KB)")
        
        if show_progress:
            print(f"    [DONE] Directory upload complete")
        
        return f"azblob://{self.account_name}/{self.container_name}/{blob_prefix}"
    
    def download_file(
        self,
        blob_name: str,
        local_path: str,
        show_progress: bool = True,
    ) -> str:
        """
        Download a file from Azure Blob Storage.
        
        Args:
            blob_name: Blob name in container
            local_path: Destination local path
            show_progress: Show download progress
            
        Returns:
            Local path to downloaded file
        """
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        if show_progress:
            print(f"    [DOWNLOAD] Downloading {blob_name} from Azure Blob...")
        
        blob_client = self.service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        with open(local_path, "wb") as f:
            blob_data = blob_client.download_blob()
            blob_data.readinto(f)
        
        if show_progress:
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"    [DONE] Downloaded {file_size_mb:.1f} MB")
        
        return local_path
    
    def delete_blob(
        self,
        blob_name: str,
        show_progress: bool = True,
    ) -> None:
        """
        Delete a blob from Azure Blob Storage.
        
        Args:
            blob_name: Blob name to delete
            show_progress: Show deletion progress
        """
        if show_progress:
            print(f"    [DELETE] Deleting {blob_name}...")
        
        blob_client = self.service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        blob_client.delete_blob()
        
        if show_progress:
            print(f"    [DONE] Deleted")
    
    def list_blobs(
        self,
        prefix: str = None,
    ) -> List[str]:
        """
        List blobs in container.
        
        Args:
            prefix: Optional prefix to filter blobs
            
        Returns:
            List of blob names
        """
        container_client = self.service_client.get_container_client(self.container_name)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        return [blob.name for blob in blobs]
    
    def get_blob_set(
        self,
        prefix: str = None,
    ) -> set:
        """
        Get a set of all blob names in container for efficient membership testing.
        
        Args:
            prefix: Optional prefix to filter blobs
            
        Returns:
            Set of blob names
        """
        return set(self.list_blobs(prefix=prefix))


def stage_slides_to_blob(
    slides_manifest: Dict[str, str],
    account_name: str,
    account_key: str,
    container_name: str = "mussel-staging",
    blob_prefix: str = "slides",
) -> Dict[str, str]:
    """
    Stage multiple slides to Azure Blob Storage.
    
    Args:
        slides_manifest: Dict mapping slide_id to local/S3 path
        account_name: Azure Storage account name
        account_key: Azure Storage account key
        container_name: Blob container name
        blob_prefix: Prefix for blob names
        
    Returns:
        Dict mapping slide_id to azblob:// URL
    """
    staging = AzureBlobStaging(
        account_name=account_name,
        account_key=account_key,
        container_name=container_name,
    )
    
    staged_urls = {}
    
    for slide_id, slide_path in slides_manifest.items():
        # Determine blob name
        filename = os.path.basename(slide_path)
        blob_name = f"{blob_prefix}/{filename}" if blob_prefix else filename
        
        # Upload
        print(f"Staging {slide_id}: {slide_path}")
        azblob_url = staging.upload_file(
            local_path=slide_path,
            blob_name=blob_name,
        )
        staged_urls[slide_id] = azblob_url
    
    return staged_urls


def stage_models_to_blob(
    models_dir: str,
    account_name: str,
    account_key: str,
    container_name: str = "mussel-staging",
    blob_prefix: str = "models",
) -> Dict[str, str]:
    """
    Stage model files to Azure Blob Storage.
    
    Args:
        models_dir: Local directory containing models
        account_name: Azure Storage account name
        account_key: Azure Storage account key
        container_name: Blob container name
        blob_prefix: Prefix for blob names
        
    Returns:
        Dict mapping model filename to azblob:// URL
    """
    staging = AzureBlobStaging(
        account_name=account_name,
        account_key=account_key,
        container_name=container_name,
    )
    
    staged_urls = {}
    models_path = Path(models_dir)
    
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Find all model files (recursively)
    for model_file in models_path.rglob("*"):
        if model_file.is_file():
            # Preserve directory structure in blob name
            rel_path = model_file.relative_to(models_path)
            blob_name = f"{blob_prefix}/{rel_path}" if blob_prefix else str(rel_path)
            blob_name = blob_name.replace("\\", "/")  # Normalize path separators
            
            print(f"Staging model: {rel_path}")
            azblob_url = staging.upload_file(
                local_path=str(model_file),
                blob_name=blob_name,
            )
            staged_urls[str(rel_path)] = azblob_url
    
    return staged_urls
