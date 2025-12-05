#!/usr/bin/env python3
"""
Azure Batch job submission script for tessellate-extract-features.

This script submits one or more tessellate-extract-features tasks to Azure Batch.
It handles pool creation, job creation, task submission, and monitoring.

Requirements:
    - azure-batch
    - azure-storage-blob
    - azure-identity

Install with: pip install azure-batch azure-storage-blob azure-identity
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Function to load environment variables from a file
def load_env_file(env_file_path, verbose=True):
    """
    Load environment variables from a file.
    
    Args:
        env_file_path: Path to the .env file
        verbose: Print loading messages (default: True)
        
    Returns:
        int: Number of variables loaded
    """
    if not os.path.exists(env_file_path):
        if verbose:
            print(f"Warning: {env_file_path} not found")
        return 0
    
    if verbose:
        print(f"Loading environment variables from: {env_file_path}")
    
    import re
    count = 0
    with open(env_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:]
            match = re.match(r'^([^=]+)=(.*)$', line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                os.environ[key] = value
                count += 1
    
    if verbose:
        print(f"✓ Loaded {count} environment variables")
    
    return count

# Import model pre-download utility and Azure Files staging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
try:
    from model_predownload import pre_download_models, upload_models_to_s3
except ImportError:
    print(
        "WARNING: Could not import model_predownload module. Pre-download features will be unavailable."
    )
    pre_download_models = None
    upload_models_to_s3 = None

try:
    from azure_files_staging import AzureFilesStaging, stage_models_to_azure_files
except ImportError:
    print(
        "WARNING: Could not import azure_files_staging module. Azure Files model staging features will be unavailable."
    )
    AzureFilesStaging = None
    stage_models_to_azure_files = None

try:
    from azure_blob_staging import AzureBlobStaging
except ImportError:
    print(
        "WARNING: Could not import azure_blob_staging module. Azure Blob staging features will be unavailable."
    )
    AzureBlobStaging = None

try:
    from config_loader import load_config, load_config_defaults, add_config_to_metadata
except ImportError:
    print(
        "WARNING: Could not import config_loader module. YAML config support will be unavailable."
    )
    load_config = None
    load_config_defaults = None
    add_config_to_metadata = None

try:
    from azure.batch import BatchServiceClient
    from azure.batch.batch_auth import SharedKeyCredentials
    from azure.batch import models as batchmodels
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
except ImportError:
    print("ERROR: Azure SDK packages not installed.")
    print("Install with: pip install azure-batch azure-storage-blob azure-identity")
    sys.exit(1)


class AzureBatchJobSubmitter:
    """
    Handles Azure Batch job submission for Mussel tessellate-extract-features.

    This class provides methods to:
    - Create and manage Azure Batch compute pools
    - Create jobs and submit tasks
    - Submit single tasks or batch tasks from configuration files
    - Monitor task progress and completion
    - Clean up resources (jobs and pools)

    Typical usage:
        submitter = AzureBatchJobSubmitter(account_name, account_key, account_url)
        submitter.create_pool(pool_id="my-pool", vm_size="Standard_NC6s_v3")
        submitter.create_job(job_id="my-job", pool_id="my-pool")
        submitter.submit_task(job_id="my-job", task_id="task-1", ...)
        submitter.monitor_tasks(job_id="my-job")
    """

    @staticmethod
    def log(message):
        """Simple logging helper."""
        print(f"  {message}")

    @staticmethod
    def convert_azblob_to_fsspec_url(url):
        """Convert azblob://container/path to az://container/path format for fsspec.
        
        Supports two formats:
        - azblob://container/path -> az://container/path (simple format)
        - azblob://account/container/path -> az://container/path (full format)
        """
        if url and url.startswith("azblob://"):
            remainder = url.replace("azblob://", "")
            parts = remainder.split("/")
            
            # Check if first part looks like account name (has dots or is storage account)
            if parts and "." in parts[0]:
                # Full format: azblob://account.blob.core.windows.net/container/path
                # Skip account, keep container and path
                if len(parts) > 1:
                    return f"az://{'/'.join(parts[1:])}"
                else:
                    return url  # Invalid format
            else:
                # Simple format: azblob://container/path
                # Keep everything as-is
                return f"az://{remainder}"
        return url

    # Azure GPU VM family prefixes
    GPU_VM_PREFIXES = ["Standard_NC", "Standard_ND", "Standard_NV"]

    def __init__(
        self,
        batch_account_name: str,
        batch_account_key: str,
        batch_account_url: str,
        storage_account_name: Optional[str] = None,
        storage_account_key: Optional[str] = None,
        azure_files_share_name: Optional[str] = None,
        staging_container: Optional[str] = None,
    ):
        """Initialize Azure Batch client."""
        credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
        self.batch_client = BatchServiceClient(credentials, batch_account_url)

        self.storage_account_name = storage_account_name
        self.storage_account_key = storage_account_key
        self.azure_files_share_name = azure_files_share_name
        self.staging_container = staging_container

        # Track task metadata for failure reporting
        self.task_metadata = {}

        # Track staged files for cleanup
        self.staged_files = []

        if storage_account_name and storage_account_key:
            self.blob_client = BlobServiceClient(
                account_url=f"https://{storage_account_name}.blob.core.windows.net",
                credential=storage_account_key,
            )
        else:
            self.blob_client = None

        # Initialize Azure Files staging client if configured
        if (
            storage_account_name
            and storage_account_key
            and azure_files_share_name
            and AzureFilesStaging
        ):
            self.azure_files_staging = AzureFilesStaging(
                account_name=storage_account_name,
                account_key=storage_account_key,
                share_name=azure_files_share_name,
            )
        else:
            self.azure_files_staging = None

        # Initialize Azure Blob staging client if configured
        if (
            storage_account_name
            and storage_account_key
            and staging_container
            and AzureBlobStaging
        ):
            self.azure_blob_staging = AzureBlobStaging(
                account_name=storage_account_name,
                account_key=storage_account_key,
                container_name=staging_container,
            )
        else:
            self.azure_blob_staging = None
        
        # Ensure log container exists for automatic log upload
        self._ensure_log_container()

    def _ensure_log_container(self):
        """Ensure the batch-logs container exists for automatic log upload."""
        if self.blob_client:
            try:
                container_client = self.blob_client.get_container_client("batch-logs")
                container_client.create_container()
                print("✓ Created batch-logs container for automatic log upload")
            except Exception as e:
                # Container might already exist
                if "ContainerAlreadyExists" in str(e) or "already exists" in str(e).lower():
                    pass  # This is fine
                else:
                    print(f"Warning: Could not create batch-logs container: {e}")
                    print("  Automatic log upload may not work")

    def _ensure_output_container(self, output_prefix: str):
        """Ensure the output container exists for remote output storage."""
        if not output_prefix or not self.blob_client:
            return
        
        # Parse container name from output prefix
        # Formats: 
        # - azblob://container/path (simple format)
        # - azblob://account/container/path or azblob://account.blob.core.windows.net/container/path (full format)
        # - az://container/path (fsspec format)
        container_name = None
        if output_prefix.startswith("azblob://"):
            remainder = output_prefix.replace("azblob://", "")
            parts = remainder.split("/")
            
            # Check if first part looks like account name (has dots)
            if parts and "." in parts[0]:
                # Full format: azblob://account.blob.core.windows.net/container/path
                if len(parts) >= 2:
                    container_name = parts[1]
            else:
                # Simple format: azblob://container/path
                if len(parts) >= 1:
                    container_name = parts[0]
        elif output_prefix.startswith("az://"):
            # az://container/path
            parts = output_prefix.replace("az://", "").split("/")
            if len(parts) >= 1:
                container_name = parts[0]
        
        if container_name:
            try:
                container_client = self.blob_client.get_container_client(container_name)
                container_client.create_container()
                print(f"✓ Created output container: {container_name}")
            except Exception as e:
                # Container might already exist
                if "ContainerAlreadyExists" in str(e) or "already exists" in str(e).lower():
                    print(f"[Azure Blob] Using existing container: {container_name}")
                else:
                    print(f"Warning: Could not create output container '{container_name}': {e}")
                    print("  Remote output upload may not work")

    def _should_set_intermediate_h5_path(
        self, aggregation_method: Optional[str]
    ) -> bool:
        """
        Determine if intermediate_h5_path should be set based on aggregation method.

        Intermediate files are needed for aggregation methods that operate on tile-level features:
        - "mean", "max", "model" require intermediate files
        - "identity" does not require intermediate files (direct pass-through)
        - None/unspecified does not require intermediate files

        Args:
            aggregation_method: The aggregation method to check

        Returns:
            True if intermediate_h5_path should be set, False otherwise
        """
        return aggregation_method is not None and aggregation_method != "identity"

    def _should_use_batch_encoding(self, **kwargs) -> bool:
        """Determine if we should use batch slide encoding optimization.

        Batch encoding is beneficial when processing multiple slides because:
        1. Patch encoder model is loaded once instead of N times
        2. Better GPU utilization through batched processing
        3. If using slide-level aggregation, slide encoder is also loaded once

        Batching provides benefits for:
        - Tile/patch-level feature extraction (always beneficial)
        - Slide-level aggregation with model (additional benefit)

        Note: For Azure Batch, batch encoding optimization is implemented at the
        task script level (run_tessellate_extract_features.sh) by passing multiple
        slides to a single task when beneficial.

        Returns True for any multi-slide processing scenario.
        """
        # Batch encoding is beneficial whenever processing multiple slides
        # The CLI will handle both patch extraction and slide aggregation efficiently
        return True

    def create_pool(
        self,
        pool_id: str,
        vm_size: str = "Standard_NC24ads_A100_v4",
        node_count: int = 1,
        container_image: str = "mskmind/mussel:latest-torch-gpu",
        use_gpu: bool = True,
        mount_azure_files: bool = False,
        enable_auto_scale: bool = False,
        min_node_count: Optional[int] = None,
        max_node_count: Optional[int] = None,
        auto_scale_evaluation_interval: int = 15,
        publisher: str = "microsoft-azure-batch",
        offer: str = "ubuntu-server-container",
        sku: str = "20-04-lts",
        node_agent_sku_id: str = "batch.node.ubuntu 20.04",
        container_registry_server: Optional[str] = None,
        container_registry_username: Optional[str] = None,
        container_registry_password: Optional[str] = None,
        use_container_prepull: bool = False,
        use_spot_nodes: bool = False,
        spot_node_count: Optional[int] = None,
        model_cache_blob_prefix: Optional[str] = None,
    ) -> None:
        """Create a pool of compute nodes with optional Azure Files mount and auto-scaling.

        Args:
            pool_id: Unique identifier for the pool
            vm_size: Azure VM size (e.g., Standard_NC24ads_A100_v4 for A100 GPU)
            node_count: Number of dedicated VMs in the pool (used as initial target for auto-scale or fixed count)
            container_image: Docker image to use
            use_gpu: Whether GPU support is intended (used for validation and logging)
            mount_azure_files: Whether to mount Azure Files share
            enable_auto_scale: Enable auto-scaling based on pending tasks
            min_node_count: Minimum number of nodes for auto-scaling (defaults to node_count)
            max_node_count: Maximum number of nodes for auto-scaling (required if enable_auto_scale=True)
            auto_scale_evaluation_interval: Auto-scale evaluation interval in minutes (default: 15)
            publisher: Azure VM image publisher (default: microsoft-azure-batch)
            offer: Azure VM image offer (default: ubuntu-server-container)
            sku: Azure VM image SKU (default: 20-04-lts)
            node_agent_sku_id: Node agent SKU ID (default: batch.node.ubuntu 20.04)
            container_registry_server: Container registry server (e.g., myregistry.azurecr.io)
            container_registry_username: Container registry username
            container_registry_password: Container registry password
            use_container_prepull: Whether to use container pre-pull with registry (default: False)
            use_spot_nodes: Whether to use low-priority/spot nodes instead of dedicated nodes (default: False)
            spot_node_count: Number of low-priority/spot nodes (defaults to node_count if use_spot_nodes=True)
        """
        print(f"Creating pool '{pool_id}'...")

        # Default spot_node_count to node_count if using spot nodes
        if use_spot_nodes and spot_node_count is None:
            spot_node_count = node_count

        # Validate auto-scale configuration
        if enable_auto_scale:
            if max_node_count is None:
                raise ValueError(
                    "max_node_count is required when enable_auto_scale is True"
                )
            if min_node_count is None:
                min_node_count = node_count
            if min_node_count > max_node_count:
                raise ValueError(
                    f"min_node_count ({min_node_count}) cannot be greater than max_node_count ({max_node_count})"
                )
            if use_spot_nodes:
                print(
                    f"  Auto-scaling: Enabled with low-priority/spot nodes (min: {min_node_count}, max: {max_node_count} nodes)"
                )
            else:
                print(
                    f"  Auto-scaling: Enabled with dedicated nodes (min: {min_node_count}, max: {max_node_count} nodes)"
                )
        else:
            if use_spot_nodes:
                print(f"  Low-priority/spot nodes: {spot_node_count} (fixed)")
            else:
                print(f"  Dedicated nodes: {node_count} (fixed)")

        # Validate GPU configuration
        is_gpu_vm = any(vm_size.startswith(prefix) for prefix in self.GPU_VM_PREFIXES)

        if use_gpu and not is_gpu_vm:
            print(
                f"  WARNING: GPU support requested but VM size '{vm_size}' does not appear to be a GPU-enabled VM"
            )
            print(
                f"  GPU-enabled VM sizes typically start with: {', '.join(self.GPU_VM_PREFIXES)}"
            )
        elif not use_gpu and is_gpu_vm:
            print(
                f"  NOTE: GPU support disabled but VM size '{vm_size}' appears to be GPU-enabled"
            )

        if use_gpu:
            print(f"  GPU support: Enabled (VM size: {vm_size})")
        else:
            print(f"  GPU support: Disabled (VM size: {vm_size})")

        # Container configuration for pre-pulling from ACR
        container_config = None
        if use_container_prepull:
            print(f"  Container pre-pull: Enabled (image: {container_image})")
            
            # Configure container registries if credentials provided
            container_registries = []
            if container_registry_server and container_registry_username and container_registry_password:
                print(f"  Container registry: {container_registry_server}")
                container_registries.append(
                    batchmodels.ContainerRegistry(
                        registry_server=container_registry_server,
                        user_name=container_registry_username,
                        password=container_registry_password,
                    )
                )
            
            # Pre-fetch the container image at pool creation time
            container_config = batchmodels.ContainerConfiguration(
                type='dockerCompatible',
                container_image_names=[container_image],
                container_registries=container_registries if container_registries else None,
            )
        else:
            print(f"  Container pre-pull: Disabled (image pulled in start task)")

        # VM configuration
        image_ref = batchmodels.ImageReference(
            publisher=publisher,
            offer=offer,
            sku=sku,
            version="latest",
        )

        vm_config = batchmodels.VirtualMachineConfiguration(
            image_reference=image_ref,
            node_agent_sku_id=node_agent_sku_id,
            container_configuration=container_config,
        )

        # Start task to configure node for docker GPU access
        # For ubuntu-hpc images (needed for A100 Gen2 support), install NVIDIA drivers
        # For ubuntu-server-container images, drivers are pre-installed
        # Stage models from Azure Files to persistent cache if Azure Files is configured
        if self.azure_files_staging:
            model_download_cmd = """# Copy models from Azure Files to persistent cache
                echo '[MODEL_CACHE] Staging models from Azure Files to persistent cache...'
                # Wait for Azure Files mount to be ready
                echo '[MODEL_CACHE] Waiting for Azure Files mount...'
                for i in $(seq 1 30); do
                    if [ -d /mnt/batch/tasks/fsmounts/azfiles ]; then
                        echo '[MODEL_CACHE] Azure Files mount is ready'
                        break
                    fi
                    echo "[MODEL_CACHE] Waiting for mount... attempt $$i/30"
                    sleep 2
                done
                
                if [ -d /mnt/batch/tasks/fsmounts/azfiles/models ]; then
                    echo '[MODEL_CACHE] Source: /mnt/batch/tasks/fsmounts/azfiles/models/'
                    echo '[MODEL_CACHE] Destination: /mnt/batch/tasks/cache/'
                    echo '[MODEL_CACHE] Source size:'
                    du -sh /mnt/batch/tasks/fsmounts/azfiles/models/ || echo '[MODEL_CACHE] Cannot determine size'
                    echo '[MODEL_CACHE] Available models in Azure Files:'
                    ls -1 /mnt/batch/tasks/fsmounts/azfiles/models/ 2>/dev/null | sed 's/^/[MODEL_CACHE]   - /' || echo '[MODEL_CACHE] No files visible yet'
                    echo '[MODEL_CACHE] Starting rsync...'
                    if rsync -av /mnt/batch/tasks/fsmounts/azfiles/models/ /mnt/batch/tasks/cache/ > /tmp/rsync.log 2>&1; then
                        echo '[MODEL_CACHE] ✓ Models copied successfully'
                        echo '[MODEL_CACHE] Cache directory contents:'
                        ls -lh /mnt/batch/tasks/cache/ | sed 's/^/[MODEL_CACHE]   /'
                        echo '[MODEL_CACHE] Total cache size:'
                        du -sh /mnt/batch/tasks/cache/
                    else
                        echo '[MODEL_CACHE] ✗ ERROR: rsync failed, check /tmp/rsync.log'
                        cat /tmp/rsync.log
                        # Don't exit 1, allow tasks to download models on-demand
                        echo '[MODEL_CACHE] Continuing anyway, models will be downloaded on-demand'
                    fi
                else
                    echo '[MODEL_CACHE] No models directory found in Azure Files mount (/mnt/batch/tasks/fsmounts/azfiles/models)'
                    echo '[MODEL_CACHE] Models will be downloaded on-demand'
                fi
"""
        else:
            # Models will be downloaded on-demand from Hugging Face to /mnt/batch/tasks/cache
            model_download_cmd = ""
        
        if "ubuntu-hpc" in offer:
            # ubuntu-hpc images come with NVIDIA drivers and nvidia-docker2 pre-installed
            # Conditionally pull the image if not using pre-pull
            pull_image_cmd = "" if use_container_prepull else f"echo 'Pulling container image to /mnt/docker...'\n                docker pull {container_image}\n                "
            
            start_task_cmd = f'''/bin/bash -c "
                set -e
                echo 'Setting up Docker for GPU support...'
                
                # Check if Docker is already configured correctly
                NEEDS_CONFIG=false
                if [ ! -f /etc/docker/daemon.json ]; then
                    echo 'Docker daemon.json does not exist'
                    NEEDS_CONFIG=true
                elif ! grep -q \\"/mnt/docker\\" /etc/docker/daemon.json; then
                    echo 'Docker data-root not set to /mnt/docker'
                    NEEDS_CONFIG=true
                elif ! grep -q \\"nvidia\\" /etc/docker/daemon.json; then
                    echo 'Docker nvidia runtime not configured'
                    NEEDS_CONFIG=true
                fi
                
                if [ "$$NEEDS_CONFIG" = "true" ]; then
                    echo 'Configuring Docker to use temporary disk...'
                    mkdir -p /mnt/docker
                    systemctl stop docker || true
                    cat > /etc/docker/daemon.json << 'DOCKEREOF'
{{
  \\"data-root\\": \\"/mnt/docker\\",
  \\"default-runtime\\": \\"nvidia\\",
  \\"runtimes\\": {{
    \\"nvidia\\": {{
      \\"path\\": \\"nvidia-container-runtime\\",
      \\"runtimeArgs\\": []
    }}
  }}
}}
DOCKEREOF
                    systemctl daemon-reload
                    systemctl start docker || true
                else
                    echo 'Docker already configured correctly'
                    # Reset failed state if Docker is in a failed state
                    if ! systemctl is-active --quiet docker; then
                        echo 'Docker not running, resetting and starting...'
                        systemctl reset-failed docker || true
                        systemctl start docker || true
                    fi
                fi
                
                # Wait for Docker to be ready
                echo 'Waiting for Docker to be ready...'
                for i in $(seq 1 30); do
                    if docker info >/dev/null 2>&1; then
                        echo 'Docker is ready!'
                        break
                    fi
                    echo \\"Waiting for Docker... attempt $$i/30\\"
                    sleep 2
                done
                
                # Verify Docker is actually working
                if ! docker info >/dev/null 2>&1; then
                    echo 'ERROR: Docker failed to start properly'
                    exit 1
                fi
                
                echo \\"Docker data root: $(docker info | grep 'Docker Root Dir')\\"
                
                # Create cache directories
                # Persistent model cache (survives across tasks)
                mkdir -p /mnt/batch/tasks/cache
                chmod -R 777 /mnt/batch/tasks/cache
                # Temporary working directories
                mkdir -p /mnt/batch/tasks/workitems/tmp
                chmod -R 777 /mnt/batch/tasks/workitems
                
                # Setup automatic cleanup (from scripts/azure_batch/setup_automatic_cleanup.sh)
                echo 'Setting up automatic cleanup cron job...'
                cat > /opt/cleanup_batch.sh << 'CLEANUP_SCRIPT_END'
#!/bin/bash
# Automatic cleanup - runs daily
/usr/bin/find /mnt/batch/tasks/workitems/tmp -type f -mtime +1 -delete 2>/dev/null || true
/usr/bin/find /mnt/batch/tasks/workitems/tmp -type d -name 'mussel_work_*' -mtime +1 -delete 2>/dev/null || true
/usr/bin/find /mnt/batch/tasks/workitems/tmp -type d -name '*wandb*' -mtime +1 -delete 2>/dev/null || true
/usr/bin/find /mnt/batch/tasks/workitems/tmp -type d -name 'pymp-*' -mtime +1 -delete 2>/dev/null || true
/usr/bin/find /tmp -type f -mtime +1 -delete 2>/dev/null || true
echo \\"[$(date)] Automatic cleanup completed\\" >> /var/log/batch-cleanup.log
CLEANUP_SCRIPT_END
                chmod +x /opt/cleanup_batch.sh
                (crontab -l 2>/dev/null || true; echo \\"0 2 * * * /opt/cleanup_batch.sh\\") | crontab -
                /opt/cleanup_batch.sh
                echo 'Automatic cleanup configured - runs daily at 2:00 AM'
                
                {model_download_cmd}
                {pull_image_cmd}
                # Add batch user to docker group
                usermod -aG docker _azbatch
                
                echo 'Docker GPU setup complete!'
"'''
        else:
            # For images with pre-installed drivers, conditionally pull image
            pull_image_cmd = "" if use_container_prepull else f"docker pull {container_image}\n                "
            
            start_task_cmd = f'''/bin/bash -c "
                mkdir -p /mnt/docker
                systemctl stop docker || true
                cat > /etc/docker/daemon.json << 'DOCKEREOF'
{{
  \\"data-root\\": \\"/mnt/docker\\"
}}
DOCKEREOF
                # Restart Docker - may show error but will recover
                systemctl start docker || systemctl restart docker || true
                
                # Wait for Docker to be ready
                for i in $(seq 1 30); do
                    if docker info >/dev/null 2>&1; then
                        break
                    fi
                    sleep 2
                done
                
                # Verify Docker is working
                if ! docker info >/dev/null 2>&1; then
                    exit 1
                fi
                
                # Create cache directories
                mkdir -p /mnt/batch/tasks/cache
                chmod -R 777 /mnt/batch/tasks/cache
                mkdir -p /mnt/batch/tasks/workitems/tmp
                chmod -R 777 /mnt/batch/tasks/workitems
                
                # Setup automatic cleanup (from scripts/azure_batch/setup_automatic_cleanup.sh)
                echo 'Setting up automatic cleanup cron job...'
                cat > /opt/cleanup_batch.sh << 'CLEANUP_SCRIPT_END'
#!/bin/bash
# Automatic cleanup - runs daily
/usr/bin/find /mnt/batch/tasks/workitems/tmp -type f -mtime +1 -delete 2>/dev/null || true
/usr/bin/find /mnt/batch/tasks/workitems/tmp -type d -name 'mussel_work_*' -mtime +1 -delete 2>/dev/null || true
/usr/bin/find /mnt/batch/tasks/workitems/tmp -type d -name '*wandb*' -mtime +1 -delete 2>/dev/null || true
/usr/bin/find /mnt/batch/tasks/workitems/tmp -type d -name 'pymp-*' -mtime +1 -delete 2>/dev/null || true
/usr/bin/find /tmp -type f -mtime +1 -delete 2>/dev/null || true
echo \\"[$(date)] Automatic cleanup completed\\" >> /var/log/batch-cleanup.log
CLEANUP_SCRIPT_END
                chmod +x /opt/cleanup_batch.sh
                (crontab -l 2>/dev/null || true; echo \\"0 2 * * * /opt/cleanup_batch.sh\\") | crontab -
                /opt/cleanup_batch.sh
                echo 'Automatic cleanup configured - runs daily at 2:00 AM'
                
                {model_download_cmd}
                {pull_image_cmd}
                usermod -aG docker _azbatch
"'''

        start_task = batchmodels.StartTask(
            command_line=start_task_cmd,
            user_identity=batchmodels.UserIdentity(
                auto_user=batchmodels.AutoUserSpecification(
                    scope=batchmodels.AutoUserScope.pool,
                    elevation_level=batchmodels.ElevationLevel.admin,
                )
            ),
            wait_for_success=True,
            max_task_retry_count=2,
        )

        # Add Azure Files mount configuration if Azure Files share is specified
        # Azure Files is used for model staging only (not slides)
        mount_config = None
        if self.azure_files_share_name:
            print(
                f"  Configuring Azure Files mount: share '{self.azure_files_share_name}' (for models)"
            )
            mount_config = [
                batchmodels.MountConfiguration(
                    azure_file_share_configuration=batchmodels.AzureFileShareConfiguration(
                        account_name=self.storage_account_name,
                        azure_file_url=f"https://{self.storage_account_name}.file.core.windows.net/{self.azure_files_share_name}",
                        account_key=self.storage_account_key,
                        relative_mount_path="azfiles",
                        mount_options="-o vers=3.0,dir_mode=0777,file_mode=0777,sec=ntlmssp,cache=strict,actimeo=3600,mfsymlinks",
                    )
                )
            ]

        # Pool configuration
        if enable_auto_scale:
            # Auto-scale formula based on pending tasks with unusable node handling
            # Based on: https://learn.microsoft.com/en-us/answers/questions/1699080/azure-batch-better-handle-unusable-nodes
            if use_spot_nodes:
                # Use low-priority nodes for auto-scaling
                # Simplified formula that works on pool startup
                auto_scale_formula = f"""
                    minNodes = {min_node_count};
                    maxNodes = {max_node_count};
                    
                    // Get count of pending tasks
                    pendingTaskCount = $PendingTasks.GetSample(1);
                    
                    // Target nodes = max(minNodes, pending tasks) capped at maxNodes
                    $TargetLowPriorityNodes = min(maxNodes, max(minNodes, pendingTaskCount));
                    $TargetDedicatedNodes = 0;
                    $NodeDeallocationOption = taskcompletion;
                """
            else:
                # Use dedicated nodes for auto-scaling
                # Simplified formula that works on pool startup
                auto_scale_formula = f"""
                    minNodes = {min_node_count};
                    maxNodes = {max_node_count};
                    
                    // Get count of pending tasks
                    pendingTaskCount = $PendingTasks.GetSample(1);
                    
                    // Target nodes = max(minNodes, pending tasks) capped at maxNodes
                    $TargetDedicatedNodes = min(maxNodes, max(minNodes, pendingTaskCount));
                    $TargetLowPriorityNodes = 0;
                    $NodeDeallocationOption = taskcompletion;
                """

            pool = batchmodels.PoolAddParameter(
                id=pool_id,
                virtual_machine_configuration=vm_config,
                vm_size=vm_size,
                enable_auto_scale=True,
                auto_scale_formula=auto_scale_formula,
                auto_scale_evaluation_interval=datetime.timedelta(
                    minutes=auto_scale_evaluation_interval
                ),
                mount_configuration=mount_config,
                start_task=start_task,  # Configure docker permissions and pre-pull image
            )
        else:
            # Fixed pool size
            if use_spot_nodes:
                # Use low-priority nodes only
                pool = batchmodels.PoolAddParameter(
                    id=pool_id,
                    virtual_machine_configuration=vm_config,
                    vm_size=vm_size,
                    target_dedicated_nodes=0,
                    target_low_priority_nodes=spot_node_count,
                    enable_auto_scale=False,
                    mount_configuration=mount_config,
                    start_task=start_task,  # Configure docker permissions and pre-pull image
                )
            else:
                # Use dedicated nodes only
                pool = batchmodels.PoolAddParameter(
                    id=pool_id,
                    virtual_machine_configuration=vm_config,
                    vm_size=vm_size,
                    target_dedicated_nodes=node_count,
                    enable_auto_scale=False,
                    mount_configuration=mount_config,
                    start_task=start_task,  # Configure docker permissions and pre-pull image
                )

        try:
            self.batch_client.pool.add(pool)
            print(f"Pool '{pool_id}' created successfully")
            if mount_config:
                print(
                    f"  Azure Files share '{self.azure_files_share_name}' will be mounted at /mnt/batch/tasks/fsmounts/azfiles"
                )
        except batchmodels.BatchErrorException as e:
            if e.error.code == "PoolExists":
                print(f"Pool '{pool_id}' already exists")
            else:
                raise

    def create_job(self, job_id: str, pool_id: str, delete_pool_on_completion: bool = False) -> None:
        """Create a job in the specified pool.
        
        Args:
            job_id: Unique identifier for the job
            pool_id: ID of the pool to run the job on
            delete_pool_on_completion: If True, configure job to delete pool when all tasks complete
        """
        print(f"Creating job '{job_id}'...")

        # Configure job to terminate when all tasks complete if auto-deletion is requested
        # Note: Pool deletion must be done separately via delete_pool() or --delete-pool with --monitor
        job_params = {
            'id': job_id,
            'pool_info': batchmodels.PoolInformation(pool_id=pool_id),
        }
        
        if delete_pool_on_completion:
            # Terminate job when all tasks complete
            # This allows monitoring scripts to detect completion and trigger pool deletion
            job_params['on_all_tasks_complete'] = batchmodels.OnAllTasksComplete.terminate_job
            print(f"  Auto-termination: Enabled (job will terminate when all tasks complete)")
            print(f"  Note: Use --monitor --delete-pool to automatically delete pool after job completes")
        
        job = batchmodels.JobAddParameter(**job_params)

        try:
            self.batch_client.job.add(job)
            print(f"Job '{job_id}' created successfully")
        except batchmodels.BatchErrorException as e:
            if e.error.code == "JobExists":
                print(f"Job '{job_id}' already exists")
            else:
                raise

    def _create_task(
        self,
        job_id: str,
        task_id: str,
        slide_path: str = None,
        slide_paths: Optional[List[str]] = None,
        slide_ids: Optional[List[str]] = None,
        output_dir_for_batch: Optional[str] = None,
        intermediate_h5_path: Optional[str] = None,
        aggregation_method: str = "identity",
        slide_model_types: Optional[str] = None,
        slide_batch_size: int = 8,
        classifier_pkl: Optional[str] = None,
        classifier_threshold: float = 0.75,
        prefilter_model_type: Optional[str] = None,
        model_types: Optional[str] = None,
        model_dir: Optional[str] = None,
        seg_config_group: Optional[str] = None,
        segment_threshold: Optional[int] = None,
        patch_size: Optional[int] = None,
        step_size: Optional[int] = None,
        mpp: Optional[float] = None,
        seg_level: Optional[int] = None,
        segment_max_value: Optional[int] = None,
        median_blur_ksize: Optional[int] = None,
        morphology_ex_kernel: Optional[int] = None,
        ref_patch_size: Optional[int] = None,
        use_otsu: Optional[bool] = None,
        tissue_area_threshold: Optional[int] = None,
        hole_area_threshold: Optional[int] = None,
        max_num_holes: Optional[int] = None,
        num_workers: int = 4,
        batch_size: int = 64,
        model_batch_sizes: Optional[Dict[str, int]] = None,
        use_gpu: bool = True,
        keep_intermediate_files: bool = False,
        hf_token: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_endpoint_url: Optional[str] = None,
        max_retry_count: int = 3,
        container_image: str = "mskmind/mussel:latest-torch-gpu",
        cleanup_staged_file: bool = False,
        use_container_prepull: bool = False,
        script_blob_url: Optional[str] = None,
    ) -> batchmodels.TaskAddParameter:
        """
        Create a task object for Azure Batch (does not submit).
        
        Returns a TaskAddParameter object that can be submitted via task.add() or task.add_collection().
        """

        # Build environment variables
        env_vars = []

        # Always use batch processing mode (even for single slides)
        # This simplifies the code and ensures consistent output staging
        if slide_paths:
            env_vars.extend(
                [
                    batchmodels.EnvironmentSetting(
                        name="SLIDE_PATHS", value=",".join(slide_paths)
                    ),
                    batchmodels.EnvironmentSetting(
                        name="SLIDE_BATCH_SIZE", value=str(slide_batch_size)
                    ),
                ]
            )
            if slide_ids:
                env_vars.append(
                    batchmodels.EnvironmentSetting(
                        name="SLIDE_IDS", value=",".join(slide_ids)
                    )
                )
        elif slide_path:
            # Convert single slide_path to list for consistency
            env_vars.extend(
                [
                    batchmodels.EnvironmentSetting(
                        name="SLIDE_PATHS", value=slide_path
                    ),
                    batchmodels.EnvironmentSetting(
                        name="SLIDE_BATCH_SIZE", value=str(slide_batch_size)
                    ),
                ]
            )

        # Common environment variables
        common_env_vars = [
            batchmodels.EnvironmentSetting(
                name="NUM_WORKERS", value=str(num_workers)
            ),
            batchmodels.EnvironmentSetting(
                name="BATCH_SIZE", value=str(batch_size)
            ),
            batchmodels.EnvironmentSetting(
                name="USE_GPU", value=str(use_gpu).lower()
            ),
            batchmodels.EnvironmentSetting(
                name="KEEP_INTERMEDIATE_FILES",
                value=str(keep_intermediate_files).lower(),
            ),
            batchmodels.EnvironmentSetting(
                name="AGGREGATION_METHOD", value=aggregation_method
            ),
        ]
        
        # Add model_batch_sizes as JSON string if provided
        if model_batch_sizes:
            common_env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="MODEL_BATCH_SIZES", value=json.dumps(model_batch_sizes)
                )
            )
        
        # Only set prefilter model type if provided (singular - only one prefilter model)
        if prefilter_model_type:
            common_env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="PREFILTER_MODEL_TYPE", value=prefilter_model_type
                )
            )
        
        env_vars.extend(common_env_vars)

        # SegConfig group or individual parameters
        if seg_config_group:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="SEG_CONFIG_GROUP", value=seg_config_group
                )
            )

        # Individual SegConfig parameters (only set if provided)
        if segment_threshold is not None:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="SEGMENT_THRESHOLD", value=str(segment_threshold)
                )
            )
        if patch_size is not None:
            env_vars.append(
                batchmodels.EnvironmentSetting(name="PATCH_SIZE", value=str(patch_size))
            )
        if step_size is not None:
            env_vars.append(
                batchmodels.EnvironmentSetting(name="STEP_SIZE", value=str(step_size))
            )
        if mpp is not None:
            env_vars.append(batchmodels.EnvironmentSetting(name="MPP", value=str(mpp)))
        if seg_level is not None:
            env_vars.append(
                batchmodels.EnvironmentSetting(name="SEG_LEVEL", value=str(seg_level))
            )
        if segment_max_value is not None:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="SEGMENT_MAX_VALUE", value=str(segment_max_value)
                )
            )
        if median_blur_ksize is not None:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="MEDIAN_BLUR_KSIZE", value=str(median_blur_ksize)
                )
            )
        if morphology_ex_kernel is not None:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="MORPHOLOGY_EX_KERNEL", value=str(morphology_ex_kernel)
                )
            )
        if ref_patch_size is not None:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="REF_PATCH_SIZE", value=str(ref_patch_size)
                )
            )
        if use_otsu is not None:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="USE_OTSU", value=str(use_otsu).lower()
                )
            )
        if tissue_area_threshold is not None:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="TISSUE_AREA_THRESHOLD", value=str(tissue_area_threshold)
                )
            )
        if hole_area_threshold is not None:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="HOLE_AREA_THRESHOLD", value=str(hole_area_threshold)
                )
            )
        if max_num_holes is not None:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="MAX_NUM_HOLES", value=str(max_num_holes)
                )
            )

        if intermediate_h5_path:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="INTERMEDIATE_H5_PATH", value=intermediate_h5_path
                )
            )

        # Handle slide models - comma-separated list
        if slide_model_types:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="SLIDE_MODEL_TYPES", value=slide_model_types
                )
            )

        if classifier_pkl:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="CLASSIFIER_PKL", value=classifier_pkl
                )
            )
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="CLASSIFIER_THRESHOLD", value=str(classifier_threshold)
                )
            )

        # Handle model types - comma-separated list
        if model_types:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="MODEL_TYPES", value=model_types
                )
            )

        # Define persistent cache directory
        cache_dir = "/mnt/batch/tasks/cache"
        
        # Always set MODEL_DIR to the persistent cache directory
        env_vars.append(
            batchmodels.EnvironmentSetting(
                name="MODEL_DIR", value=cache_dir
            )
        )
        
        # If model_dir is provided and Azure Files is configured, set flag to rsync from Azure Files
        # Otherwise, models will be downloaded on-demand from Hugging Face
        if model_dir and self.azure_files_staging:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="RSYNC_MODELS_FROM_AZURE_FILES", value="true"
                )
            )
        
        env_vars.extend([
            # Set HuggingFace cache environment variables
            batchmodels.EnvironmentSetting(
                name="HF_HOME", value=cache_dir
            ),
            batchmodels.EnvironmentSetting(
                name="TRANSFORMERS_CACHE", value=cache_dir
            ),
            batchmodels.EnvironmentSetting(
                name="HF_HUB_CACHE", value=f"{cache_dir}/hub"
            ),
            batchmodels.EnvironmentSetting(
                name="TORCH_HOME", value=cache_dir
            ),
            # Set temp directory to batch tasks directory (has more space)
            batchmodels.EnvironmentSetting(
                name="TMPDIR", value="/mnt/batch/tasks/workitems/tmp"
            ),
            # Other helpful environment variables
            batchmodels.EnvironmentSetting(
                name="PYTHONUNBUFFERED", value="1"
            ),
            batchmodels.EnvironmentSetting(
                name="OMP_NUM_THREADS", value="1"
            ),
        ])

        if hf_token:
            env_vars.append(
                batchmodels.EnvironmentSetting(name="HF_TOKEN", value=hf_token)
            )

        # Add AWS credentials if provided (for S3 access)
        if aws_access_key_id:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="AWS_ACCESS_KEY_ID", value=aws_access_key_id
                )
            )
        if aws_secret_access_key:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="AWS_SECRET_ACCESS_KEY", value=aws_secret_access_key
                )
            )
        if aws_region:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="AWS_DEFAULT_REGION", value=aws_region
                )
            )
        if aws_endpoint_url:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="AWS_ENDPOINT_URL", value=aws_endpoint_url
                )
            )

        # Add Azure Files cleanup settings if enabled
        if cleanup_staged_file:
            env_vars.append(
                batchmodels.EnvironmentSetting(name="CLEANUP_STAGED_FILE", value="true")
            )

        # Always add Azure Storage credentials if available (needed for input and output staging)
        # Set both naming conventions:
        # - AZURE_STORAGE_ACCOUNT/KEY for az CLI (input staging)
        # - AZURE_STORAGE_ACCOUNT_NAME/ACCOUNT_KEY for adlfs/fsspec (output writing)
        if self.storage_account_name:
            env_vars.extend([
                batchmodels.EnvironmentSetting(
                    name="AZURE_STORAGE_ACCOUNT", value=self.storage_account_name
                ),
                batchmodels.EnvironmentSetting(
                    name="AZURE_STORAGE_ACCOUNT_NAME", value=self.storage_account_name
                )
            ])
        if self.storage_account_key:
            env_vars.extend([
                batchmodels.EnvironmentSetting(
                    name="AZURE_STORAGE_KEY", value=self.storage_account_key
                ),
                batchmodels.EnvironmentSetting(
                    name="AZURE_STORAGE_ACCOUNT_KEY", value=self.storage_account_key
                )
            ])
        if self.azure_files_share_name:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="AZURE_FILES_SHARE", value=self.azure_files_share_name
                )
            )
        
        # Add script blob URL if provided (allows updating scripts without rebuilding container)
        if script_blob_url:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="SCRIPT_BLOB_URL", value=script_blob_url
                )
            )

        # Convert env vars to docker -e format
        docker_env_args = " ".join([f'-e {var.name}="{var.value}"' for var in env_vars])

        # Task command - invoke docker directly with GPU support
        # Mount Azure Files if available, otherwise use local paths
        azure_files_mount = (
            "/mnt/batch/tasks/fsmounts/azfiles" if self.azure_files_staging else ""
        )

        # Performance optimization: Use local VM storage for caching and temp files
        # - /mnt/batch/tasks: Azure Batch task directory (automatically mounted, no need for explicit mounts)
        # - /mnt/batch/tasks/cache: Persistent model cache directory (survives across tasks)
        # - /mnt/batch/tasks/workitems: Task working directory
        # No explicit volume mounts needed - /mnt/batch/tasks is automatically available
        volume_mounts = (
            f"-v {azure_files_mount}:{azure_files_mount}" if azure_files_mount else ""
        )
        volume_mounts += " -v /mnt/batch/tasks:/mnt/batch/tasks"

        task_command = f'/bin/bash -c "mkdir -p /mnt/batch/tasks/cache /mnt/batch/tasks/workitems/tmp && chmod -R 777 /mnt/batch/tasks && docker run --rm --user root --ipc host --gpus all --shm-size=8g {docker_env_args} {volume_mounts} {container_image} /app/scripts/azure_batch/run_tessellate_extract_features.sh"'

        # Task constraints with retry configuration and retention policy
        # Set retention time to automatically delete task directory after completion
        # This prevents disk space exhaustion from accumulated task directories
        # Use 0 minutes to delete immediately after upload completes
        from datetime import timedelta as dt_timedelta
        task_constraints = batchmodels.TaskConstraints(
            max_task_retry_count=max_retry_count,
            retention_time=dt_timedelta(minutes=0)  # Delete task directory immediately after completion and upload
        )

        # Configure automatic output file staging to blob storage
        # This replaces manual upload logic in bash scripts
        output_files = []
        if self.storage_account_name and self.storage_account_key and output_dir_for_batch:
            # Create SAS URL for uploads (valid for 7 days)
            from datetime import datetime, timedelta
            from azure.storage.blob import generate_container_sas, ContainerSasPermissions
            
            # Extract container and path from output_dir_for_batch (azblob://container/path or az://container/path format)
            if output_dir_for_batch.startswith("azblob://"):
                parts = output_dir_for_batch.replace("azblob://", "").split("/", 1)
                output_container = parts[0]
                output_path_prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
            elif output_dir_for_batch.startswith("az://"):
                parts = output_dir_for_batch.replace("az://", "").split("/", 1)
                output_container = parts[0]
                output_path_prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
            else:
                # Skip output file staging for non-Azure paths
                output_container = None
                output_path_prefix = None
                
            if output_container:
                # Ensure output container exists
                try:
                    from azure.storage.blob import BlobServiceClient
                    blob_service_client = BlobServiceClient(
                        account_url=f"https://{self.storage_account_name}.blob.core.windows.net",
                        credential=self.storage_account_key
                    )
                    container_client = blob_service_client.get_container_client(output_container)
                    if not container_client.exists():
                        print(f"Creating output container: {output_container}")
                        container_client.create_container()
                except Exception as e:
                    print(f"Warning: Could not ensure output container exists: {e}")
                
                # Create SAS token for output container
                sas_token = generate_container_sas(
                    account_name=self.storage_account_name,
                    container_name=output_container,
                    account_key=self.storage_account_key,
                    permission=ContainerSasPermissions(write=True, create=True, list=True),
                    expiry=datetime.utcnow() + timedelta(days=7)
                )
                
                output_container_url = f"https://{self.storage_account_name}.blob.core.windows.net/{output_container}?{sas_token}"
                
                # Upload all result files (h5, pt, json) on task completion
                # Use $AZ_BATCH_TASK_WORKING_DIR for task working directory (where output/ is created)
                output_files.append(
                    batchmodels.OutputFile(
                        file_pattern='$AZ_BATCH_TASK_WORKING_DIR/output/**/*',
                        destination=batchmodels.OutputFileDestination(
                            container=batchmodels.OutputFileBlobContainerDestination(
                                container_url=output_container_url,
                                path=output_path_prefix
                            )
                        ),
                        upload_options=batchmodels.OutputFileUploadOptions(
                            upload_condition=batchmodels.OutputFileUploadCondition.task_completion
                        )
                    )
                )
            
            # Upload logs to a dedicated container
            log_container = "batch-logs"
            # Ensure log container exists
            try:
                from azure.storage.blob import BlobServiceClient
                blob_service_client = BlobServiceClient(
                    account_url=f"https://{self.storage_account_name}.blob.core.windows.net",
                    credential=self.storage_account_key
                )
                container_client = blob_service_client.get_container_client(log_container)
                if not container_client.exists():
                    print(f"Creating log container: {log_container}")
                    container_client.create_container()
            except Exception as e:
                print(f"Warning: Could not ensure log container exists: {e}")
            
            sas_token = generate_container_sas(
                account_name=self.storage_account_name,
                container_name=log_container,
                account_key=self.storage_account_key,
                permission=ContainerSasPermissions(write=True, create=True, list=True),
                expiry=datetime.utcnow() + timedelta(days=7)
            )
            
            log_container_url = f"https://{self.storage_account_name}.blob.core.windows.net/{log_container}?{sas_token}"
            
            # Upload stdout - for containers it's in ../stdout.txt
            output_files.append(
                batchmodels.OutputFile(
                    file_pattern='../stdout.txt',
                    destination=batchmodels.OutputFileDestination(
                        container=batchmodels.OutputFileBlobContainerDestination(
                            container_url=log_container_url,
                            path=f"{job_id}/{task_id}/stdout.txt"
                        )
                    ),
                    upload_options=batchmodels.OutputFileUploadOptions(
                        upload_condition=batchmodels.OutputFileUploadCondition.task_completion
                    )
                )
            )
            
            # Upload stderr - for containers it's in ../stderr.txt
            output_files.append(
                batchmodels.OutputFile(
                    file_pattern='../stderr.txt',
                    destination=batchmodels.OutputFileDestination(
                        container=batchmodels.OutputFileBlobContainerDestination(
                            container_url=log_container_url,
                            path=f"{job_id}/{task_id}/stderr.txt"
                        )
                    ),
                    upload_options=batchmodels.OutputFileUploadOptions(
                        upload_condition=batchmodels.OutputFileUploadCondition.task_completion
                    )
                )
            )

        # Create task with container settings if prepull is enabled
        if use_container_prepull:
            # Use TaskContainerSettings for container-enabled pools
            # Download the latest wrapper script from Azure Blob (allows testing without rebuilding Docker image)
            # The entrypoint is /bin/bash, so command_line should start with -c
            if hasattr(self, 'script_blob_url') and self.script_blob_url:
                # Parse storage account and container from script_blob_url
                # Format: https://<account>.blob.core.windows.net/<container>/scripts/...
                import re
                match = re.match(r'https://([^.]+)\.blob\.core\.windows\.net/([^/]+)/', self.script_blob_url)
                if match:
                    script_storage_account = match.group(1)
                    script_container = match.group(2)
                    blob_name = "scripts/azure_batch/run_tessellate_extract_features.sh"
                    # Download script using credentials from the parsed storage account
                    # Use az CLI to download (handles auth via env vars AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_KEY)
                    container_command = (
                        f'-c "az storage blob download '
                        f'--account-name {script_storage_account} '
                        f'--container-name {script_container} '
                        f'--name {blob_name} '
                        f'--file /tmp/run_wrapper.sh '
                        f'--overwrite '
                        f'&& chmod +x /tmp/run_wrapper.sh && /tmp/run_wrapper.sh"'
                    )
                else:
                    # Fallback to baked-in script if URL parsing fails
                    container_command = '-c /app/scripts/azure_batch/run_tessellate_extract_features.sh'
            else:
                # Fallback to baked-in script
                container_command = '-c /app/scripts/azure_batch/run_tessellate_extract_features.sh'
            
            # Azure Batch container settings
            # Note: Volume mounts are handled by Azure Batch pool configuration, not container_run_options
            # The /mnt/batch/tasks directory is automatically available in containers
            # Use minimal container_run_options - Azure Batch has restrictions on allowed flags
            container_run_options = '--rm --user=root --ipc=host --shm-size=8g --entrypoint=/bin/bash'
            
            container_settings = batchmodels.TaskContainerSettings(
                image_name=container_image,
                container_run_options=container_run_options,
                working_directory='taskWorkingDirectory'  # Use Azure Batch task directory to access mounts
            )
            
            task = batchmodels.TaskAddParameter(
                id=task_id,
                command_line=container_command,
                environment_settings=env_vars,  # Pass env vars directly to container
                constraints=task_constraints,
                container_settings=container_settings,
                output_files=output_files if output_files else None
                # Note: user_identity with admin elevation cannot be used with container_settings
                # Using --user=root in container_run_options instead
            )
        else:
            # Use docker invocation for non-container pools
            task = batchmodels.TaskAddParameter(
                id=task_id,
                command_line=task_command,
                environment_settings=[],  # Env vars passed via docker -e
                constraints=task_constraints,
                output_files=output_files if output_files else None,
            )

        # Return the task object (caller decides whether to submit via add or add_collection)
        return task
    
    def submit_task(
        self,
        job_id: str,
        task_id: str,
        slide_path: str = None,
        slide_paths: Optional[List[str]] = None,
        slide_ids: Optional[List[str]] = None,
        output_dir_for_batch: Optional[str] = None,
        intermediate_h5_path: Optional[str] = None,
        aggregation_method: str = "identity",
        slide_model_types: Optional[str] = None,
        slide_batch_size: int = 8,
        classifier_pkl: Optional[str] = None,
        classifier_threshold: float = 0.75,
        prefilter_model_type: Optional[str] = None,
        model_types: Optional[str] = None,
        model_dir: Optional[str] = None,
        seg_config_group: Optional[str] = None,
        segment_threshold: Optional[int] = None,
        patch_size: Optional[int] = None,
        step_size: Optional[int] = None,
        mpp: Optional[float] = None,
        seg_level: Optional[int] = None,
        segment_max_value: Optional[int] = None,
        median_blur_ksize: Optional[int] = None,
        morphology_ex_kernel: Optional[int] = None,
        ref_patch_size: Optional[int] = None,
        use_otsu: Optional[bool] = None,
        tissue_area_threshold: Optional[int] = None,
        hole_area_threshold: Optional[int] = None,
        max_num_holes: Optional[int] = None,
        num_workers: int = 4,
        batch_size: int = 64,
        model_batch_sizes: Optional[Dict[str, int]] = None,
        use_gpu: bool = True,
        keep_intermediate_files: bool = False,
        hf_token: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_endpoint_url: Optional[str] = None,
        max_retry_count: int = 3,
        container_image: str = "mskmind/mussel:latest-torch-gpu",
        cleanup_staged_file: bool = False,
        use_container_prepull: bool = False,
        script_blob_url: Optional[str] = None,
    ) -> None:
        """
        Submit a tessellate-extract-features task to Azure Batch.
        
        Wrapper around _create_task that creates and submits a task.
        """
        print(f"Submitting task '{task_id}' to job '{job_id}'...")
        
        # Create task object
        task = self._create_task(
            job_id=job_id,
            task_id=task_id,
            slide_path=slide_path,
            slide_paths=slide_paths,
            slide_ids=slide_ids,
            output_dir_for_batch=output_dir_for_batch,
            intermediate_h5_path=intermediate_h5_path,
            aggregation_method=aggregation_method,
            slide_model_types=slide_model_types,
            slide_batch_size=slide_batch_size,
            classifier_pkl=classifier_pkl,
            classifier_threshold=classifier_threshold,
            prefilter_model_type=prefilter_model_type,
            model_types=model_types,
            model_dir=model_dir,
            seg_config_group=seg_config_group,
            segment_threshold=segment_threshold,
            patch_size=patch_size,
            step_size=step_size,
            mpp=mpp,
            seg_level=seg_level,
            segment_max_value=segment_max_value,
            median_blur_ksize=median_blur_ksize,
            morphology_ex_kernel=morphology_ex_kernel,
            ref_patch_size=ref_patch_size,
            use_otsu=use_otsu,
            tissue_area_threshold=tissue_area_threshold,
            hole_area_threshold=hole_area_threshold,
            max_num_holes=max_num_holes,
            num_workers=num_workers,
            batch_size=batch_size,
            model_batch_sizes=model_batch_sizes,
            use_gpu=use_gpu,
            keep_intermediate_files=keep_intermediate_files,
            hf_token=hf_token,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region,
            aws_endpoint_url=aws_endpoint_url,
            max_retry_count=max_retry_count,
            container_image=container_image,
            cleanup_staged_file=cleanup_staged_file,
            use_container_prepull=use_container_prepull,
            script_blob_url=script_blob_url,
        )

        try:
            self.batch_client.task.add(job_id, task)

            # Store task metadata for failure tracking and manifest generation
            self.task_metadata[task_id] = {
                "slide_path": slide_path,
                "model_types": model_types,
                "output_dir": output_dir_for_batch,
            }
            if intermediate_h5_path:
                self.task_metadata[task_id]["intermediate_h5_path"] = (
                    intermediate_h5_path
                )

            print(
                f"Task '{task_id}' submitted successfully (max retries: {max_retry_count})"
            )
        except batchmodels.BatchErrorException as e:
            # Handle task already exists (can happen with incremental submission retries)
            if 'TaskExists' in str(e):
                print(f"Task '{task_id}' already exists, skipping")
            else:
                print(f"Error submitting task: {e}")
                raise


    def submit_tasks_from_csv(
        self,
        job_id: str,
        csv_file: str,
        output_s3_prefix: Optional[str] = None,
        container_image: str = "mskmind/mussel:latest-torch-gpu",
        models: Optional[List[str]] = None,
        slide_models: Optional[List[str]] = None,
        staged_slide_paths: Optional[Dict[str, str]] = None,
        slides_per_task: int = 1,
        use_container_prepull: bool = False,
        batch_offset: int = 0,  # For incremental submission: starting batch number
        total_batches_global: Optional[int] = None,  # For incremental submission: total batches across all submissions
        **default_params,
    ) -> None:
        """
        Submit tasks from a CSV manifest file.

        CSV format:
            slide_id,slide_path
            slide_001,s3://bucket/slides/slide_001.svs
            slide_002,/local/path/slide_002.svs

        Args:
            job_id: Azure Batch job ID
            csv_file: Path to CSV manifest file
            output_s3_prefix: Output prefix for results (supports s3://, azblob://, azfiles://, https://, or local paths)
            container_image: Docker image to use
            models: List of model types to run sequentially in same task
            staged_slide_paths: Optional dict mapping slide_id to staged Azure Files paths
            slides_per_task: Number of slides to group per Azure Batch task (default: 1).
                This controls how many slides are processed together in a single batch task.
                When > 1, slides are grouped to optimize model loading (load once per batch).
                Recommended: 8-16 for better efficiency. Note: This is different from the
                'slide_batch_size' parameter which controls internal batching within a task.
            **default_params: Default parameters for all tasks (e.g., prefilter_model_type, batch_size)
        """
        print(f"Loading task manifest from '{csv_file}'...")

        # Get prefilter model types (used for directory organization when single model)
        prefilter_model = default_params.get("prefilter_model_type", None)

        # Determine which models to process - create separate tasks per model
        # to avoid staging multiple models per task
        all_models = []
        all_slide_models = []
        
        # Get patch-level models
        if models:
            all_models = models
        else:
            model_types_str = default_params.get("model_types", None)
            if model_types_str:
                all_models = [m.strip() for m in model_types_str.split(",")]
        
        # Get slide-level models
        if slide_models:
            all_slide_models = slide_models
        else:
            slide_model_types_str = default_params.get("slide_model_types", None)
            if slide_model_types_str:
                all_slide_models = [m.strip() for m in slide_model_types_str.split(",")]
        
        # Combine all models to process
        models_to_process = all_models + all_slide_models
        
        if not models_to_process:
            print("ERROR: No models specified for processing")
            return
        
        print(f"Will create separate tasks for {len(models_to_process)} models: {', '.join(models_to_process)}")
        print(f"  Patch-level models: {', '.join(all_models) if all_models else 'None'}")
        print(f"  Slide-level models: {', '.join(all_slide_models) if all_slide_models else 'None'}")

        # Extract model_dir from default_params if provided
        model_dir = default_params.get("model_dir", None)

        # Read all slides from CSV
        slides = []
        all_slides_remote = True  # Track if all slides are already remote
        staged_slide_count = 0
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle different column names flexibly
                slide_id = row.get("slide_id") or row.get("sample_id") or row.get("image_id")
                slide_path = row.get("slide_path") or row.get("svs_path") or row.get("path")
                
                if not slide_id or not slide_path:
                    raise ValueError(f"Could not find slide ID or path in CSV row: {row}")

                # Use staged path if available
                if staged_slide_paths and slide_id in staged_slide_paths:
                    slide_path = staged_slide_paths[slide_id]
                    staged_slide_count += 1

                # Check if this is a remote path
                if not slide_path.startswith(("azfiles://", "azblob://", "http://", "https://", "s3://")):
                    all_slides_remote = False

                slides.append({"slide_id": slide_id, "slide_path": slide_path})
        
        if staged_slide_count > 0:
            print(f"✓ Using {staged_slide_count} pre-staged slide paths")
        if all_slides_remote:
            print(f"✓ All {len(slides)} slides are already staged remotely - fast submission mode enabled")

        # Auto-adjust slides_per_task if using default value (1)
        # and slide-level model aggregation is enabled
        if slides_per_task == 1 and self._should_use_batch_encoding(
            **default_params
        ):
            slides_per_task = 8  # Recommended default for batch encoding
            print(f"\n[Auto-Batching] Detected slide-level model aggregation")
            print(
                f"  Automatically enabling batch processing with slides_per_task={slides_per_task}"
            )
            print(f"  (Use --slides-per-task to override)")

        # Determine if we should use batch encoding
        use_batch_encoding = (
            slides_per_task > 1
            and self._should_use_batch_encoding(**default_params)
        )

        # Process ALL models in a single task per batch of slides
        # This is more efficient than separate tasks per model
        print(f"\n{'='*80}")
        print(f"Processing {len(models_to_process)} models per task:")
        print(f"  Patch-level models: {', '.join(all_models) if all_models else 'None'}")
        print(f"  Slide-level models: {', '.join(all_slide_models) if all_slide_models else 'None'}")
        print(f"{'='*80}")
        
        # Set all models in parameters (comma-separated)
        model_params = dict(default_params)
        if all_models:
            model_params["model_types"] = ",".join(all_models)
        if all_slide_models:
            model_params["slide_model_types"] = ",".join(all_slide_models)
        
        # Pass model_batch_sizes dict if provided (instead of computing a single batch_size)
        # This allows per-model batch sizes to be used by tessellate_extract_features
        if "model_batch_sizes" in default_params and isinstance(default_params["model_batch_sizes"], dict):
            model_batch_sizes = default_params["model_batch_sizes"]
            model_params["model_batch_sizes"] = model_batch_sizes
            
            print(f"\n[Per-Model Batch Sizes]")
            for model, batch_size in model_batch_sizes.items():
                print(f"  {model}: {batch_size}")
        
        total_tasks_submitted = 0

        if use_batch_encoding:
            print(f"\n[Batch Encoding Optimization] Enabled")
            print(f"  Grouping slides into batches of {slides_per_task}")
        
            # Stage model directory to Azure Files if enabled and model_dir is specified
            local_model_dir = default_params.get("model_dir")
            if self.azure_files_staging and local_model_dir:
                self.log(f"\n[Azure Files] Staging models from: {local_model_dir}")
                
                # Create mapping of model names to expected file/directory names
                model_file_map = {
                    "CONCH1_5": ["conch1_5.pth", "conch.pth"],
                    "UNI": ["uni.pth"],
                    "UNI2": ["uni2.pth"],
                    "VIRCHOW": ["virchow.pth"],
                    "VIRCHOW2": ["virchow2.pth"],
                    "GIGAPATH": ["gigapath.pth"],
                    "OPTIMUS": ["optimus.pth"],
                    "TITAN_SLIDE": ["titan_slide", "gigapath_slide.pth"],
                    "CTRANSPATH": ["ctranspath.pth"],
                    "RESNET50": ["resnet50.pth"],
                }
                
                # Determine which models need to be uploaded
                models_to_upload = set()
                # Include prefilter model if specified
                prefilter_model = default_params.get("prefilter_model_type", None)
                models_for_upload = all_models + all_slide_models
                if prefilter_model:
                    models_for_upload.append(prefilter_model)
                
                for model in models_for_upload:
                    if model in model_file_map:
                        models_to_upload.update(model_file_map[model])
                
                self.log(f"  Models to upload: {', '.join(models_to_upload)}")
                
                # Check if model_dir is a local path
                if not local_model_dir.startswith(("s3://", "azfiles://", "azblob://", "http://", "https://")):
                    if os.path.exists(local_model_dir) and os.path.isdir(local_model_dir):
                        # Upload only the required model files/directories
                        self.log(f"  Uploading required model files...")
                        
                        # Upload both files and subdirectories that are needed
                        additional_files = ["version.txt"]
                        
                        for item_name in os.listdir(local_model_dir):
                            # Skip files/dirs not in models_to_upload
                            if item_name not in models_to_upload and item_name not in additional_files:
                                continue
                                
                            item_path = os.path.join(local_model_dir, item_name)
                            
                            if os.path.isdir(item_path):
                                # Upload subdirectory
                                self.log(f"    Uploading model dir: {item_name}")
                                remote_model_dir = f"models/{item_name}"
                                
                                # Upload the model directory
                                self.azure_files_staging.upload_directory(
                                    local_dir=item_path,
                                    remote_dir=remote_model_dir
                                )
                                self.log(f"      ✓ Uploaded to: {remote_model_dir}")
                            
                            elif os.path.isfile(item_path):
                                # Upload individual model file
                                self.log(f"    Uploading model file: {item_name}")
                                remote_file_path = f"models/{item_name}"
                                
                                # Upload the model file
                                self.azure_files_staging.upload_file(
                                    local_path=item_path,
                                    remote_path=remote_file_path,
                                    show_progress=False,
                                    skip_if_exists=True
                                )
                                self.log(f"      ✓ Uploaded to: {remote_file_path}")
                        
                        # Upload classifier_pkl if it exists in model_dir
                        classifier_pkl = default_params.get("classifier_pkl")
                        if classifier_pkl and not classifier_pkl.startswith(
                            ("s3://", "azfiles://", "azblob://", "http://", "https://")
                        ):
                            # If classifier_pkl is just a filename, look for it in model_dir
                            if not os.path.isabs(classifier_pkl) and "/" not in classifier_pkl:
                                classifier_pkl_path = os.path.join(local_model_dir, classifier_pkl)
                            else:
                                classifier_pkl_path = classifier_pkl
                            
                            if os.path.exists(classifier_pkl_path):
                                self.log(f"    Uploading classifier: {os.path.basename(classifier_pkl_path)}")
                                remote_path = f"models/{os.path.basename(classifier_pkl_path)}"
                                self.azure_files_staging.upload_file(
                                    local_path=classifier_pkl_path,
                                    remote_path=remote_path,
                                    show_progress=False,
                                    skip_if_exists=True
                                )
                                # Update classifier_pkl to just the filename (will be resolved in persistent cache)
                                default_params["classifier_pkl"] = os.path.basename(classifier_pkl_path)
                                self.log(f"      ✓ Uploaded to: {remote_path}")
                        
                        # Set model_dir parameter to azfiles:// URL for tasks
                        azfiles_model_dir = f"azfiles://{self.storage_account_name}/{self.azure_files_share_name}/models"
                        default_params["model_dir"] = azfiles_model_dir
                        self.log(f"\n[Azure Files] Model directory URL: {azfiles_model_dir}")
                    else:
                        self.log(f"  Warning: Model directory not found: {model_dir}")
                else:
                    # Remote path, use as-is
                    default_params["model_dir"] = model_dir
                    self.log(f"  Using remote model directory: {model_dir}")
            elif model_dir:
                # No Azure Files, pass model_dir as-is
                default_params["model_dir"] = model_dir
                self.log(f"[Model Directory] Using: {model_dir}")

            # Group slides into batches and submit in streaming fashion
            # This allows tasks to start running before all tasks are prepared
            tasks_buffer = []
            api_batch_size = 100  # Azure Batch API supports up to 100 tasks per collection
            total_batches = total_batches_global if total_batches_global else (len(slides) + slides_per_task - 1) // slides_per_task
            
            print(f"\n[Streaming Submission] Preparing and submitting {total_batches} tasks...")
            print(f"  Note: Tasks will start running as they are submitted")
            
            for batch_idx in range(0, len(slides), slides_per_task):
                    batch_slides = slides[
                        batch_idx : batch_idx + slides_per_task
                    ]

                    # Create batch task ID
                    # Use global batch number for incremental submission
                    batch_num = batch_offset + (batch_idx // slides_per_task + 1)
                    
                    # Create concise model identifier for task ID
                    # Use first model + count if multiple models
                    if len(models_to_process) == 1:
                        models_str = models_to_process[0]
                    elif len(models_to_process) == 2:
                        models_str = "_".join(models_to_process)
                    else:
                        # Multiple models: show first + count (e.g., "OPTIMUS_plus4more")
                        models_str = f"{models_to_process[0]}_plus{len(models_to_process)-1}more"
                    
                    batch_id = f"batch_{batch_num}_of_{total_batches}_{models_str}"

                    # Extract slide IDs and paths for this batch
                    slide_ids = [s["slide_id"] for s in batch_slides]
                    slide_paths_batch = [s["slide_path"] for s in batch_slides]

                    # For pre-staged slides (azblob://), skip file checks
                    # Only process local files that need staging to blob
                    if not all_slides_remote and self.azure_blob_staging:
                        staged_paths = []
                        for slide_path in slide_paths_batch:
                            # Already a remote path (azblob://, or http://) - use as-is
                            if slide_path.startswith(("azblob://", "http://", "https://", "s3://")):
                                staged_paths.append(slide_path)
                            # Local file - needs staging to blob
                            else:
                                # Only check file existence for local files
                                if os.path.exists(slide_path):
                                    slide_filename = os.path.basename(slide_path)
                            
                                    # Stage to Azure Blob
                                    blob_name = f"slides/{slide_filename}"
                                    self.azure_blob_staging.upload_file(
                                        local_path=slide_path,
                                        blob_name=blob_name,
                                        show_progress=False,
                                    )
                                    # Convert to azblob:// URL
                                    azblob_url = f"azblob://{self.storage_account_name}/{self.staging_container}/{blob_name}"
                                    staged_paths.append(azblob_url)
                                else:
                                    self.log(f"WARNING: Local file not found: {slide_path}")
                                    staged_paths.append(slide_path)
                        slide_paths_batch = staged_paths

                    # Show progress for large submissions (every 50 tasks or at milestones)
                    if batch_num % 50 == 1 or batch_num == 1 or batch_num == total_batches or batch_num % 100 == 0:
                        print(f"  Preparing task {batch_num}/{total_batches}: {batch_id}")

                    # Create task object
                    # output_dir_for_batch should be the remote Azure blob path for output file staging
                    # The local path is always hardcoded to "output" in the CLI command
                    task_obj = self._create_task(
                        job_id=job_id,
                        task_id=batch_id,
                        slide_paths=slide_paths_batch,
                        slide_ids=slide_ids,
                        output_dir_for_batch=output_s3_prefix,
                        slide_batch_size=model_params.get("slide_batch_size", 8),
                        aggregation_method=model_params.get(
                            "aggregation_method", "identity"
                        ),
                        classifier_pkl=model_params.get("classifier_pkl"),
                        classifier_threshold=model_params.get(
                            "classifier_threshold", 0.75
                        ),
                        prefilter_model_type=model_params.get(
                            "prefilter_model_type", None
                        ),
                        model_types=model_params.get("model_types"),
                        model_dir=model_params.get("model_dir"),
                        slide_model_types=model_params.get("slide_model_types"),
                        seg_config_group=model_params.get("seg_config_group"),
                        segment_threshold=model_params.get("segment_threshold"),
                        patch_size=model_params.get("patch_size"),
                        step_size=model_params.get("step_size"),
                        mpp=model_params.get("mpp"),
                        seg_level=model_params.get("seg_level"),
                        segment_max_value=model_params.get("segment_max_value"),
                        median_blur_ksize=model_params.get("median_blur_ksize"),
                        morphology_ex_kernel=model_params.get("morphology_ex_kernel"),
                        ref_patch_size=model_params.get("ref_patch_size"),
                        use_otsu=model_params.get("use_otsu"),
                        tissue_area_threshold=model_params.get("tissue_area_threshold"),
                        hole_area_threshold=model_params.get("hole_area_threshold"),
                        max_num_holes=model_params.get("max_num_holes"),
                        num_workers=model_params.get("num_workers", 4),
                        batch_size=model_params.get("batch_size", 64),
                        model_batch_sizes=model_params.get("model_batch_sizes"),
                        use_gpu=model_params.get("use_gpu", True),
                        keep_intermediate_files=model_params.get(
                            "keep_intermediate_files", False
                        ),
                        hf_token=model_params.get("hf_token"),
                        aws_access_key_id=model_params.get("aws_access_key_id"),
                        aws_secret_access_key=model_params.get("aws_secret_access_key"),
                        aws_region=model_params.get("aws_region"),
                        aws_endpoint_url=model_params.get("aws_endpoint_url"),
                        max_retry_count=model_params.get("max_retry_count", 3),
                        container_image=container_image,
                        use_container_prepull=use_container_prepull,
                        script_blob_url=model_params.get("script_blob_url"),
                    )
                    
                    tasks_buffer.append(task_obj)

                    # Store task configuration in metadata (excluding secrets)
                    if add_config_to_metadata:
                        add_config_to_metadata(self.task_metadata, model_params, batch_id)
                
                    total_tasks_submitted += 1
                    
                    # Submit when buffer reaches API batch size
                    if len(tasks_buffer) >= api_batch_size:
                        progress_pct = int((total_tasks_submitted / total_batches) * 100)
                        print(f"  [{progress_pct}%] Submitting tasks {total_tasks_submitted - len(tasks_buffer) + 1} to {total_tasks_submitted}...")
                        try:
                            result = self.batch_client.task.add_collection(job_id, tasks_buffer)
                            
                            # Check for errors
                            if hasattr(result, 'value') and result.value:
                                for task_result in result.value:
                                    if task_result.status != batchmodels.TaskAddStatus.success:
                                        # Suppress TaskExists warnings (expected with incremental submission)
                                        if task_result.error and 'TaskExists' in str(task_result.error.code):
                                            continue  # Skip TaskExists warnings
                                        print(f"    WARNING: Task {task_result.task_id} failed to add: {task_result.error}")
                            
                            print(f"    ✓ Submitted {len(tasks_buffer)} tasks (tasks can now start running)")
                            tasks_buffer = []
                        except Exception as e:
                            print(f"    ERROR: Failed to submit batch: {e}")
                            # Fall back to individual submission
                            for task in tasks_buffer:
                                try:
                                    self.batch_client.task.add(job_id, task)
                                except Exception as task_error:
                                    # Skip TaskExists errors (expected with retries)
                                    if 'TaskExists' not in str(task_error):
                                        print(f"      ERROR: Failed to submit task {task.id}: {task_error}")
                            tasks_buffer = []

            # Submit remaining tasks in buffer
            if tasks_buffer:
                print(f"  [100%] Submitting final {len(tasks_buffer)} tasks...")
                try:
                    result = self.batch_client.task.add_collection(job_id, tasks_buffer)
                    
                    # Check for errors
                    if hasattr(result, 'value') and result.value:
                        for task_result in result.value:
                            if task_result.status != batchmodels.TaskAddStatus.success:
                                # Suppress TaskExists warnings (expected with incremental submission)
                                if task_result.error and 'TaskExists' in str(task_result.error.code):
                                    continue  # Skip TaskExists warnings
                                print(f"    WARNING: Task {task_result.task_id} failed to add: {task_result.error}")
                    
                    print(f"    ✓ Submitted {len(tasks_buffer)} tasks (tasks can now start running)")
                except Exception as e:
                    print(f"    ERROR: Failed to submit batch: {e}")
                    # Fall back to individual submission
                    for task in tasks_buffer:
                        try:
                            self.batch_client.task.add(job_id, task)
                        except Exception as task_error:
                            # Skip TaskExists errors (expected with retries)
                            if 'TaskExists' not in str(task_error):
                                print(f"      ERROR: Failed to submit task {task.id}: {task_error}")
            
            print(f"[Streaming Submission] Complete - {total_tasks_submitted} tasks submitted")
            # End of batch loop
            
        print(f"\n{'='*80}")
        print(f"Submitted {total_tasks_submitted} total tasks")
        print(f"  Models per task: {len(models_to_process)}")
        print(f"  Slides per task: {slides_per_task}")
        print(f"{'='*80}")
        return

    def monitor_tasks(self, job_id: str, poll_interval: int = 30, delete_pool_on_completion: bool = False, pool_id: Optional[str] = None) -> None:
        """Monitor task progress.
        
        Args:
            job_id: ID of the job to monitor
            poll_interval: Time in seconds between status checks
            delete_pool_on_completion: If True, delete the pool when all tasks complete
            pool_id: Pool ID to delete (required if delete_pool_on_completion is True)
        """
        print(f"Monitoring tasks in job '{job_id}'...")
        print("Press Ctrl+C to stop monitoring (tasks will continue running)")
        
        if delete_pool_on_completion and not pool_id:
            print("WARNING: delete_pool_on_completion is True but no pool_id provided. Pool will not be deleted.")
            delete_pool_on_completion = False

        try:
            while True:
                tasks = list(self.batch_client.task.list(job_id))

                total = len(tasks)
                completed = sum(
                    1 for t in tasks if t.state == batchmodels.TaskState.completed
                )
                active = sum(
                    1 for t in tasks if t.state == batchmodels.TaskState.active
                )
                running = sum(
                    1 for t in tasks if t.state == batchmodels.TaskState.running
                )

                print(
                    f"\n[{datetime.datetime.now()}] Status: {completed}/{total} completed, "
                    f"{running} running, {active} active"
                )

                for task in tasks:
                    print(f"  - {task.id}: {task.state}")

                if completed == total:
                    print("\nAll tasks completed!")
                    
                    # Delete pool if requested
                    if delete_pool_on_completion and pool_id:
                        print(f"\nDeleting pool '{pool_id}' as all tasks have completed...")
                        try:
                            self.delete_pool(pool_id)
                            print(f"Pool '{pool_id}' deleted successfully")
                        except Exception as e:
                            print(f"ERROR: Failed to delete pool '{pool_id}': {e}")
                    
                    break

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\nStopped monitoring (tasks still running)")

    def save_failed_tasks(
        self,
        job_id: str,
        output_file: str,
        task_metadata: Optional[Dict[str, Dict]] = None,
    ) -> int:
        """
        Save failed tasks to a CSV file for resubmission.

        Args:
            job_id: Azure Batch job ID
            output_file: Path to save failed tasks CSV
            task_metadata: Optional dict mapping task_id to task configuration

        Returns:
            Number of failed tasks saved
        """
        print(f"Checking for failed tasks in job '{job_id}'...")

        tasks = list(self.batch_client.task.list(job_id))
        failed_tasks = []

        for task in tasks:
            # Check if task failed (completed with non-zero exit code or execution failed)
            if task.state == batchmodels.TaskState.completed:
                # Get execution info to check exit code
                if task.execution_info and task.execution_info.exit_code != 0:
                    failed_tasks.append(task)

        if not failed_tasks:
            print("No failed tasks found")
            return 0

        print(f"Found {len(failed_tasks)} failed tasks, saving to '{output_file}'...")

        # Prepare failed task data
        failed_data = []
        for task in failed_tasks:
            task_id = task.id
            task_info = {
                "task_id": task_id,
                "state": str(task.state),
                "exit_code": task.execution_info.exit_code
                if task.execution_info
                else "N/A",
            }

            # If we have metadata, include the original task configuration
            if task_metadata and task_id in task_metadata:
                task_info.update(task_metadata[task_id])
            else:
                # Extract from environment variables if available
                if task.environment_settings:
                    for env_var in task.environment_settings:
                        if env_var.name == "SLIDE_PATH":
                            task_info["slide_path"] = env_var.value

            failed_data.append(task_info)

        # Write to CSV
        if failed_data:
            # Determine CSV headers from first failed task
            headers = list(failed_data[0].keys())

            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(failed_data)

            print(f"Saved {len(failed_data)} failed tasks to '{output_file}'")
            print(f"You can resubmit these tasks using: --csv-manifest {output_file}")

        return len(failed_data)

    def generate_results_manifest(
        self,
        job_id: str,
        output_file: str,
        task_metadata: Optional[Dict[str, Dict]] = None,
    ) -> int:
        """
        Generate a manifest of successfully completed result files.

        Args:
            job_id: Azure Batch job ID
            output_file: Path to save results manifest CSV
            task_metadata: Optional dict mapping task_id to task configuration

        Returns:
            Number of successful tasks in manifest
        """
        print(f"Generating results manifest for job '{job_id}'...")

        tasks = list(self.batch_client.task.list(job_id))
        successful_tasks = []

        for task in tasks:
            # Check if task completed successfully
            if task.state == batchmodels.TaskState.completed:
                if task.execution_info and task.execution_info.exit_code == 0:
                    successful_tasks.append(task)

        if not successful_tasks:
            print("No successful tasks found")
            return 0

        print(f"Found {len(successful_tasks)} successful tasks, generating manifest...")

        # Prepare manifest data
        manifest_data = []
        for task in successful_tasks:
            task_id = task.id
            task_info = {
                "task_id": task_id,
                "state": "completed",
                "exit_code": 0,
            }

            # If we have metadata, include the output paths
            if task_metadata and task_id in task_metadata:
                metadata = task_metadata[task_id]
                task_info["slide_path"] = metadata.get("slide_path", "")
                
                # Extract slide_id from slide_path
                slide_path = metadata.get("slide_path", "")
                if slide_path:
                    # Get filename from path (handles azblob:// and s3:// URLs)
                    filename = slide_path.split('/')[-1]
                    slide_id = filename.rsplit('.', 1)[0]  # Remove extension
                    task_info["slide_id"] = slide_id
                    
                    # Generate expected output paths for each model
                    output_dir = metadata.get("output_dir", "")
                    model_types = metadata.get("model_types", "")
                    
                    if output_dir and model_types:
                        # Parse model_types (can be comma-separated string or already a list)
                        if isinstance(model_types, str):
                            models = [m.strip() for m in model_types.split(',')]
                        else:
                            models = model_types if isinstance(model_types, list) else [model_types]
                        
                        # Generate output paths for each model
                        task_info["model_types"] = ','.join(models)
                        output_paths = []
                        for model in models:
                            # Expected output format: {output_dir}/{model}/h5/{slide_id}.h5
                            h5_path = f"{output_dir}/{model}/h5/{slide_id}.h5"
                            output_paths.append(h5_path)
                        task_info["output_h5_paths"] = ';'.join(output_paths)

                # Add intermediate path if present
                if "intermediate_h5_path" in metadata:
                    task_info["intermediate_h5_path"] = metadata.get(
                        "intermediate_h5_path", ""
                    )

                # Add configuration parameters if present (excluding secrets)
                if "config" in metadata:
                    config_data = metadata["config"]
                    # Flatten configuration into task_info with 'config_' prefix
                    # Note: Nested dictionaries and lists are skipped for CSV compatibility
                    for key, value in config_data.items():
                        # Skip nested dictionaries and lists for CSV simplicity
                        if not isinstance(value, (dict, list)):
                            task_info[f"config_{key}"] = value

            else:
                # Extract from environment variables if available
                if task.environment_settings:
                    for env_var in task.environment_settings:
                        if env_var.name == "SLIDE_PATH":
                            task_info["slide_path"] = env_var.value
                        elif env_var.name == "INTERMEDIATE_H5_PATH":
                            task_info["intermediate_h5_path"] = env_var.value
                        elif env_var.name == "PREFILTER_MODEL_TYPE":
                            task_info["model_type"] = env_var.value

            manifest_data.append(task_info)

        # Write to CSV
        if manifest_data:
            # Determine CSV headers from first task
            headers = list(manifest_data[0].keys())

            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(manifest_data)

            print(
                f"Saved results manifest with {len(manifest_data)} successful tasks to '{output_file}'"
            )

        return len(manifest_data)


    def delete_job(self, job_id: str) -> None:
        """Delete a job."""
        print(f"Deleting job '{job_id}'...")
        self.batch_client.job.delete(job_id)
        print(f"Job '{job_id}' deleted")

    def delete_pool(self, pool_id: str) -> None:
        """Delete a pool."""
        print(f"Deleting pool '{pool_id}'...")
        self.batch_client.pool.delete(pool_id)
        print(f"Pool '{pool_id}' deleted")


def main():
    parser = argparse.ArgumentParser(
        description="Submit tessellate-extract-features jobs to Azure Batch"
    )

    # Environment file configuration
    parser.add_argument(
        "--env-file",
        help="Path to environment file with credentials (e.g., secrets.env). "
        "Loads Azure, AWS, and HuggingFace credentials from file.",
    )

    # Azure credentials (can be provided via command-line, environment variables, or config file)
    parser.add_argument(
        "--batch-account-name",
        help="Azure Batch account name (or set AZURE_BATCH_ACCOUNT_NAME env var)",
    )
    parser.add_argument(
        "--batch-account-key",
        help="Azure Batch account key (or set AZURE_BATCH_ACCOUNT_KEY env var)",
    )
    parser.add_argument(
        "--batch-account-url",
        help="Azure Batch account URL (or set AZURE_BATCH_ACCOUNT_URL env var)",
    )
    parser.add_argument(
        "--storage-account-name",
        help="Azure Storage account name (optional, or set AZURE_STORAGE_ACCOUNT env var)",
    )
    parser.add_argument(
        "--storage-account-key",
        help="Azure Storage account key (optional, or set AZURE_STORAGE_KEY env var)",
    )

    # Azure Files staging configuration (for models only)
    parser.add_argument(
        "--azure-files-share-name", help="Azure Files share name for staging model files"
    )
    parser.add_argument(
        "--mount-azure-files",
        action="store_true",
        help="Mount Azure Files share to batch pool nodes. When combined with --model-dir, models will be uploaded to Azure Files and mounted for faster access.",
    )

    # Azure Blob staging configuration
    parser.add_argument(
        "--staging-container",
        help="Azure Blob container name for staging slides (default: uses output_prefix). Models should be staged to Azure Files using --azure-files-share-name and --model-dir.",
    )
    parser.add_argument(
        "--azure-blob-container",
        dest="staging_container",
        help="(Deprecated) Use --staging-container instead",
    )
    parser.add_argument(
        "--stage-to-azure-blob",
        action="store_true",
        help="Stage input files to Azure Blob Storage before processing (uses incremental submission with parallel workers)",
    )
    parser.add_argument(
        "--staging-workers",
        type=int,
        default=20,
        help="Number of parallel workers for slide staging (default: 20, max: 50). Higher values speed up staging but may hit rate limits.",
    )

    # Pool configuration
    parser.add_argument("--pool-id", help="Pool ID (can be specified in config file)")
    parser.add_argument(
        "--create-pool", action="store_true", help="Create pool if it doesn't exist"
    )

    # Default values for pool parameters (used for detecting if config should override)
    DEFAULT_VM_SIZE = "Standard_NC24ads_A100_v4"
    DEFAULT_NODE_COUNT = 1
    DEFAULT_CONTAINER_IMAGE = "mskmind/mussel:latest"
    DEFAULT_AUTO_SCALE_INTERVAL = 15

    parser.add_argument(
        "--vm-size", default=DEFAULT_VM_SIZE, help="VM size for pool nodes"
    )
    parser.add_argument(
        "--node-count",
        type=int,
        default=DEFAULT_NODE_COUNT,
        help="Number of nodes in pool (or initial/min count for auto-scaling)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Enable GPU support for pool nodes (default: True)",
    )
    parser.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Disable GPU support for pool nodes",
    )
    parser.add_argument(
        "--container-image",
        default=DEFAULT_CONTAINER_IMAGE,
        help="Docker container image",
    )
    parser.add_argument(
        "--enable-auto-scale",
        action="store_true",
        help="Enable auto-scaling based on pending tasks",
    )
    parser.add_argument(
        "--min-node-count",
        type=int,
        help="Minimum number of nodes for auto-scaling (defaults to --node-count)",
    )
    parser.add_argument(
        "--max-node-count",
        type=int,
        help="Maximum number of nodes for auto-scaling (required if --enable-auto-scale)",
    )
    parser.add_argument(
        "--auto-scale-evaluation-interval",
        type=int,
        default=DEFAULT_AUTO_SCALE_INTERVAL,
        help="Auto-scale evaluation interval in minutes (default: 15)",
    )

    # VM image configuration
    parser.add_argument(
        "--publisher",
        default="microsoft-azure-batch",
        help="Azure VM image publisher (default: microsoft-azure-batch)",
    )
    parser.add_argument(
        "--offer",
        default="ubuntu-server-container",
        help="Azure VM image offer (default: ubuntu-server-container)",
    )
    parser.add_argument(
        "--sku",
        default="20-04-lts",
        help="Azure VM image reference SKU (default: 20-04-lts)",
    )
    parser.add_argument(
        "--node-agent-sku-id",
        default="batch.node.ubuntu 20.04",
        help="Node agent SKU ID (default: batch.node.ubuntu 20.04)",
    )

    # Container registry configuration
    parser.add_argument(
        "--container-registry-server",
        help="Container registry server (e.g., myregistry.azurecr.io or AZURE_CONTAINER_REGISTRY_SERVER env var)",
    )
    parser.add_argument(
        "--container-registry-username",
        help="Container registry username (or AZURE_CONTAINER_REGISTRY_USERNAME env var)",
    )
    parser.add_argument(
        "--container-registry-password",
        help="Container registry password (or AZURE_CONTAINER_REGISTRY_PASSWORD env var)",
    )

    parser.add_argument(
        "--use-spot-nodes",
        action="store_true",
        help="Use low-priority/spot nodes instead of dedicated nodes for cost savings (default: False)",
    )
    parser.add_argument(
        "--spot-node-count",
        type=int,
        help="Number of low-priority/spot nodes (defaults to --node-count if --use-spot-nodes is set)",
    )

    # Job configuration
    parser.add_argument("--job-id", help="Job ID (can be specified in config file)")
    parser.add_argument("--create-job", action="store_true", help="Create job")

    # Task configuration
    parser.add_argument(
        "--config-file",
        "--config",
        dest="config_file",
        help="Configuration file with parameters (JSON or YAML format). "
        "Can be used alone with task definitions, or with --csv-manifest to provide default parameters.",
    )
    parser.add_argument(
        "--csv-manifest",
        help="CSV manifest file with slide_id,slide_path columns. "
        "Can be used with --config-file or --config to load parameters from config.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Output prefix for results. Can be:\n"
        "  - S3: s3://bucket/results/\n"
        "  - Azure Blob: https://account.blob.core.windows.net/container/results/ or azblob://account/container/results/\n"
        "  - Azure Files: azfiles://account/share/outputs/\n"
        "  - Local path: /path/to/output\n"
        "Results are automatically uploaded to this location after processing.",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated list of model types to run (e.g., CTRANSPATH,CLIP,VIRCHOW)",
    )
    parser.add_argument(
        "--slides-per-task",
        type=int,
        default=1,
        help="Number of slides to group per Azure Batch task (default: 1, auto-adjusted to 8). "
        "Groups slides together to load models once instead of N times. Beneficial for all multi-slide processing, "
        "especially with slide-level model aggregation. Recommended: 8-16 for better efficiency. "
        "Auto-enabled (batch_size=8) when processing multiple slides; use 1 to disable. "
        "Note: Not applicable when using --stage-to-azure-blob (incremental staging). "
        "This is different from 'slide_batch_size' which controls internal batching within a task.",
    )
    parser.add_argument(
        "--slide-batch-size",
        type=int,
        default=None,
        help="Batch size for slide encoding within a task (default: 8). "
        "Controls how many slides are processed together in GPU memory during slide-level encoding. "
        "Increase for better GPU utilization, decrease if running out of memory. "
        "This is different from '--slides-per-task' which controls task grouping.",
    )
    parser.add_argument("--task-id", help="Single task ID")
    parser.add_argument(
        "--slide-path", help="Path to slide file (for single task, can be s3://)"
    )
    parser.add_argument(
        "--output-h5-path", help="Output H5 path (for single task, can be s3://)"
    )
    parser.add_argument(
        "--output-pt-path", help="Output PT path (for single task, can be s3://)"
    )

    # AWS credentials for S3 access (can be provided via command-line, environment variables, or config file)
    parser.add_argument(
        "--aws-access-key-id",
        help="AWS access key ID for S3 (or set AWS_ACCESS_KEY_ID env var)",
    )
    parser.add_argument(
        "--aws-secret-access-key",
        help="AWS secret access key for S3 (or set AWS_SECRET_ACCESS_KEY env var)",
    )
    parser.add_argument(
        "--aws-region",
        default=None,
        help="AWS region (default: us-east-1, or set AWS_DEFAULT_REGION env var)",
    )
    parser.add_argument(
        "--aws-endpoint-url",
        help="Custom S3 endpoint URL (e.g., for MinIO or Ceph, or set AWS_ENDPOINT_URL env var)",
    )

    # HuggingFace token for model downloads (can be provided via command-line, environment variables, or config file)
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token for model downloads (or set HF_TOKEN env var)",
    )

    # Model pre-download configuration
    parser.add_argument(
        "--pre-download-models",
        action="store_true",
        default=True,
        help="Pre-download models before job submission (default: True for batch jobs)",
    )
    parser.add_argument(
        "--no-pre-download-models",
        dest="pre_download_models",
        action="store_false",
        help="Disable model pre-download",
    )
    parser.add_argument(
        "--model-dir",
        default="./model_cache",
        help="Local directory containing models. Models will be downloaded here if pre-download-models is enabled, then uploaded to Azure Files if specified (default: ./model_cache)",
    )

    # Retry configuration
    parser.add_argument(
        "--max-retry-count",
        type=int,
        default=None,
        help="Maximum number of retry attempts for failed tasks (default: 3)",
    )
    parser.add_argument(
        "--save-failed-tasks", help="Save failed tasks to CSV file for resubmission"
    )
    parser.add_argument(
        "--generate-manifest",
        help="Generate manifest of successful result files to CSV",
    )

    # Monitoring and cleanup
    parser.add_argument("--monitor", action="store_true", help="Monitor task progress")
    parser.add_argument(
        "--delete-job", action="store_true", help="Delete job after completion"
    )
    parser.add_argument(
        "--delete-pool",
        action="store_true",
        help="Delete pool after completion. When used with --monitor, "
        "the pool will be deleted after all tasks complete. "
        "Otherwise, it will be deleted immediately.",
    )
    parser.add_argument(
        "--auto-delete-pool",
        action="store_true",
        help="Configure Azure Batch to automatically delete the pool when the job completes. "
        "This uses Azure Batch's native pool lifetime management.",
    )
    parser.add_argument(
        "--cleanup-staged-files",
        action="store_true",
        help="Clean up staged files in Azure Blob after job submission",
    )

    args = parser.parse_args()

    # Load environment file if specified
    if args.env_file:
        if os.path.exists(args.env_file):
            load_env_file(args.env_file, verbose=True)
        else:
            print(f"ERROR: Environment file not found: {args.env_file}")
            sys.exit(1)

    # Load config file early if provided, to check for model paths before pre-download
    config_defaults = {}
    if args.config_file and load_config_defaults:
        try:
            config_defaults = load_config_defaults(args.config_file, backend="azure")
        except Exception as e:
            print(f"WARNING: Failed to load config file: {e}")
            config_defaults = {}

    # Priority order for credentials: CLI args > Environment variables > Config file
    # This allows flexible credential management for CI/CD and local development

    # Azure Batch credentials
    if not args.batch_account_name:
        args.batch_account_name = os.environ.get(
            "AZURE_BATCH_ACCOUNT_NAME"
        ) or config_defaults.get("batch_account_name")

    if not args.batch_account_key:
        args.batch_account_key = os.environ.get(
            "AZURE_BATCH_ACCOUNT_KEY"
        ) or config_defaults.get("batch_account_key")

    if not args.batch_account_url:
        args.batch_account_url = os.environ.get(
            "AZURE_BATCH_ACCOUNT_URL"
        ) or config_defaults.get("batch_account_url")

    # Azure Storage credentials
    if not args.storage_account_name:
        args.storage_account_name = (
            os.environ.get("AZURE_STORAGE_ACCOUNT")
            or os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
            or config_defaults.get("storage_account_name")
        )

    if not args.storage_account_key:
        args.storage_account_key = (
            os.environ.get("AZURE_STORAGE_KEY")
            or os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
            or config_defaults.get("storage_account_key")
        )

    # AWS credentials for S3 access
    if not args.aws_access_key_id:
        args.aws_access_key_id = os.environ.get(
            "AWS_ACCESS_KEY_ID"
        ) or config_defaults.get("aws_access_key_id")

    if not args.aws_secret_access_key:
        args.aws_secret_access_key = os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        ) or config_defaults.get("aws_secret_access_key")

    if not args.aws_region:  # Not set via CLI
        env_region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get(
            "AWS_REGION"
        )
        if env_region:
            args.aws_region = env_region
        elif "aws_region" in config_defaults:
            args.aws_region = config_defaults["aws_region"]
        else:
            args.aws_region = "us-east-1"  # Default fallback

    if not args.aws_endpoint_url:
        args.aws_endpoint_url = os.environ.get(
            "AWS_ENDPOINT_URL"
        ) or config_defaults.get("aws_endpoint_url")

    # HuggingFace token
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or config_defaults.get("hf_token")

    # Container registry credentials
    if not args.container_registry_server:
        args.container_registry_server = os.environ.get(
            "AZURE_CONTAINER_REGISTRY_SERVER"
        ) or config_defaults.get("container_registry_server")
    
    if not args.container_registry_username:
        args.container_registry_username = os.environ.get(
            "AZURE_CONTAINER_REGISTRY_USERNAME"
        ) or config_defaults.get("container_registry_username")
    
    if not args.container_registry_password:
        args.container_registry_password = os.environ.get(
            "AZURE_CONTAINER_REGISTRY_PASSWORD"
        ) or config_defaults.get("container_registry_password")
    
    # Container prepull is always enabled
    args.use_container_prepull = True

    # Validate required credentials
    if not args.batch_account_name:
        print(
            "ERROR: --batch-account-name is required (or set AZURE_BATCH_ACCOUNT_NAME env var or specify in config file)"
        )
        sys.exit(1)

    if not args.batch_account_key:
        print(
            "ERROR: --batch-account-key is required (or set AZURE_BATCH_ACCOUNT_KEY env var or specify in config file)"
        )
        sys.exit(1)

    if not args.batch_account_url:
        print(
            "ERROR: --batch-account-url is required (or set AZURE_BATCH_ACCOUNT_URL env var or specify in config file)"
        )
        sys.exit(1)

    # Apply Azure-specific parameters from config file if provided
    # Command-line arguments take precedence over config file values
    if config_defaults:
        # Pool and job IDs
        if not args.pool_id and "pool_id" in config_defaults:
            args.pool_id = config_defaults["pool_id"]

        if not args.job_id and "job_id" in config_defaults:
            args.job_id = config_defaults["job_id"]

        # Storage account parameters
        if not args.storage_account_name and "storage_account_name" in config_defaults:
            args.storage_account_name = config_defaults["storage_account_name"]

        # Azure Files share name
        if (
            not args.azure_files_share_name
            and "azure_files_share_name" in config_defaults
        ):
            args.azure_files_share_name = config_defaults["azure_files_share_name"]

        # Azure Blob container name
        if (
            not args.staging_container
            and "staging_container" in config_defaults
        ):
            args.staging_container = config_defaults["staging_container"]

        # Boolean flags (only set if not already set via command-line)
        if not args.create_pool and config_defaults.get("create_pool"):
            args.create_pool = config_defaults["create_pool"]

        if not args.create_job and config_defaults.get("create_job"):
            args.create_job = config_defaults["create_job"]

        if not args.auto_delete_pool and config_defaults.get("auto_delete_pool"):
            args.auto_delete_pool = config_defaults["auto_delete_pool"]

        if not args.mount_azure_files and config_defaults.get("mount_azure_files"):
            args.mount_azure_files = config_defaults["mount_azure_files"]

        # stage_to_azure_blob and cleanup_staged_files were removed - these are handled differently now
        # Set defaults for removed args to avoid AttributeError
        if not hasattr(args, 'stage_to_azure_files'):
            args.stage_to_azure_files = False
        if not hasattr(args, 'cleanup_staged_files'):
            args.cleanup_staged_files = False

        # Container image - only override if the default value is still being used
        if (
            args.container_image == DEFAULT_CONTAINER_IMAGE
            and "container_image" in config_defaults
        ):
            args.container_image = config_defaults["container_image"]

        # VM size - only override if default is being used AND not explicitly passed on command line
        vm_size_explicitly_set = "--vm-size" in sys.argv
        if (
            not vm_size_explicitly_set
            and args.vm_size == DEFAULT_VM_SIZE
            and "vm_size" in config_defaults
        ):
            args.vm_size = config_defaults["vm_size"]

        # Node count - only override if default is being used
        if args.node_count == DEFAULT_NODE_COUNT and "node_count" in config_defaults:
            args.node_count = config_defaults["node_count"]

        # Auto-scaling parameters
        if not args.enable_auto_scale and config_defaults.get("enable_auto_scale"):
            args.enable_auto_scale = config_defaults["enable_auto_scale"]

        if args.min_node_count is None and "min_node_count" in config_defaults:
            args.min_node_count = config_defaults["min_node_count"]

        if args.max_node_count is None and "max_node_count" in config_defaults:
            args.max_node_count = config_defaults["max_node_count"]

        if (
            args.auto_scale_evaluation_interval == DEFAULT_AUTO_SCALE_INTERVAL
            and "auto_scale_evaluation_interval" in config_defaults
        ):
            args.auto_scale_evaluation_interval = config_defaults[
                "auto_scale_evaluation_interval"
            ]

        # VM image configuration parameters
        if args.publisher == "microsoft-azure-batch" and "publisher" in config_defaults:
            args.publisher = config_defaults["publisher"]

        if args.offer == "ubuntu-server-container" and "offer" in config_defaults:
            args.offer = config_defaults["offer"]

        if args.sku == "20-04-lts" and "sku" in config_defaults:
            args.sku = config_defaults["sku"]

        if (
            args.node_agent_sku_id == "batch.node.ubuntu 20.04"
            and "node_agent_sku_id" in config_defaults
        ):
            args.node_agent_sku_id = config_defaults["node_agent_sku_id"]

        # Container registry settings from config
        if not args.container_registry_server and "container_registry_server" in config_defaults:
            args.container_registry_server = config_defaults["container_registry_server"]
        
        if not args.container_registry_username and "container_registry_username" in config_defaults:
            args.container_registry_username = config_defaults["container_registry_username"]
        
        if not args.container_registry_password and "container_registry_password" in config_defaults:
            args.container_registry_password = config_defaults["container_registry_password"]
        
        # Container prepull is always enabled
        args.use_container_prepull = True
        
        # Spot node configuration from config
        if not args.use_spot_nodes and "use_spot_nodes" in config_defaults:
            args.use_spot_nodes = config_defaults["use_spot_nodes"]
        
        if args.spot_node_count is None and "spot_node_count" in config_defaults:
            args.spot_node_count = config_defaults["spot_node_count"]
        
        # If spot_node_count is set, automatically enable spot nodes
        if args.spot_node_count is not None and args.spot_node_count > 0:
            args.use_spot_nodes = True

        # Output prefix from config
        if not args.output_prefix and "output_prefix" in config_defaults:
            args.output_prefix = config_defaults["output_prefix"]

        # slides_per_task - override default value (1) with config if present
        if args.slides_per_task == 1 and "slides_per_task" in config_defaults:
            args.slides_per_task = config_defaults["slides_per_task"]
        
        # use_gpu - check if explicitly set via --no-gpu, otherwise use config
        # Since --use-gpu has action="store_true" with default=True, we need special handling
        # Only override if --no-gpu wasn't explicitly used and config has a value
        if "--no-gpu" not in sys.argv and "use_gpu" in config_defaults:
            args.use_gpu = config_defaults["use_gpu"]

    # Auto-generate pool_id and job_id if not provided (use same timestamp for consistency)
    if not args.pool_id or not args.job_id:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if not args.pool_id:
            args.pool_id = f"mussel-pool-{timestamp}"
            print(f"Auto-generated pool ID: {args.pool_id}")

        if not args.job_id:
            args.job_id = f"mussel-job-{timestamp}"
            print(f"Auto-generated job ID: {args.job_id}")

    # Handle model staging
    # If pre_download_models is enabled, download models to model_dir first
    # Then upload to Azure Files if azure_files_share_name is specified
    model_dir = args.model_dir
    
    if args.pre_download_models and pre_download_models:
        print(f"\n[Model Download] Downloading models to: {model_dir}")
        
        # Get list of models to download
        models_to_download = []
        if args.models:
            models_to_download.extend([m.strip() for m in args.models.split(",")])
        if config_defaults.get("models"):
            config_models = config_defaults.get("models")
            if isinstance(config_models, str):
                models_to_download.extend([m.strip() for m in config_models.split(",")])
            elif isinstance(config_models, list):
                models_to_download.extend(config_models)
        
        # Get prefilter model
        prefilter_model = (getattr(args, "prefilter_model_type", None) or 
                          config_defaults.get("prefilter_model_type"))
        if prefilter_model:
            models_to_download.append(prefilter_model)
        
        # Get slide model
        slide_model = (getattr(args, "slide_model_type", None) or 
                      config_defaults.get("slide_model_type"))
        if slide_model:
            models_to_download.append(slide_model)
        
        # Remove duplicates
        models_to_download = list(set(models_to_download))
        
        if models_to_download:
            print(f"[Model Download] Models to download: {', '.join(models_to_download)}")
            try:
                pre_download_models(
                    model_types=models_to_download,
                    output_dir=model_dir,
                    hf_token=args.hf_token
                )
                print(f"✓ Models downloaded to: {model_dir}")
            except Exception as e:
                print(f"⚠️  Warning: Model download failed: {e}")
                print("   Tasks will attempt to download models on-demand")
        else:
            print("[Model Download] No models specified, skipping pre-download")
    else:
        print(f"[Model Download] Skipping pre-download (using model-dir: {model_dir})")

    # Initialize submitter
    submitter = AzureBatchJobSubmitter(
        batch_account_name=args.batch_account_name,
        batch_account_key=args.batch_account_key,
        batch_account_url=args.batch_account_url,
        storage_account_name=args.storage_account_name,
        storage_account_key=args.storage_account_key,
        azure_files_share_name=args.azure_files_share_name,
        staging_container=args.staging_container,
    )

    # Model staging removed - using persistent cache at /mnt/batch/tasks/cache instead
    # Models will be downloaded on-demand from Hugging Face to the persistent cache
    pool_model_cache_prefix = None

    # Create pool if requested
    if args.create_pool:
        submitter.create_pool(
            pool_id=args.pool_id,
            vm_size=args.vm_size,
            node_count=args.node_count,
            container_image=args.container_image,
            use_gpu=args.use_gpu,
            mount_azure_files=args.mount_azure_files,
            enable_auto_scale=args.enable_auto_scale,
            min_node_count=args.min_node_count,
            max_node_count=args.max_node_count,
            auto_scale_evaluation_interval=args.auto_scale_evaluation_interval,
            publisher=args.publisher,
            offer=args.offer,
            sku=args.sku,
            node_agent_sku_id=args.node_agent_sku_id,
            container_registry_server=args.container_registry_server,
            container_registry_username=args.container_registry_username,
            container_registry_password=args.container_registry_password,
            use_container_prepull=args.use_container_prepull,
            use_spot_nodes=args.use_spot_nodes,
            spot_node_count=args.spot_node_count,
            model_cache_blob_prefix=pool_model_cache_prefix,
        )

    # Create job if requested
    if args.create_job:
        submitter.create_job(
            job_id=args.job_id, 
            pool_id=args.pool_id,
            delete_pool_on_completion=args.auto_delete_pool
        )

    # Submit tasks
    # Handle two cases:
    # 1. CSV manifest (with or without config file for parameters)
    # 2. Single task submission

    if args.csv_manifest:
        # CSV manifest (with or without config file for parameters)

        # Prepare default parameters for CSV tasks
        # config_defaults is already loaded above
        default_params = {}

        # If config file is provided, use the already-loaded defaults
        if args.config_file and config_defaults:
            print(f"Loading default parameters from config file: {args.config_file}")
            default_params.update(config_defaults)
            print(f"Loaded {len(config_defaults)} default parameters from config file")
            
            # prefilter_model_type is always singular (only one prefilter model)
            if "prefilter_model_type" in default_params:
                print(f"[Config] Using prefilter_model_type: {default_params['prefilter_model_type']}")

        # Command-line arguments override config file defaults
        if args.aws_access_key_id:
            default_params["aws_access_key_id"] = args.aws_access_key_id
        if args.aws_secret_access_key:
            default_params["aws_secret_access_key"] = args.aws_secret_access_key
        if args.aws_region is not None:
            default_params["aws_region"] = args.aws_region
        if args.hf_token:
            default_params["hf_token"] = args.hf_token
        if args.max_retry_count is not None:
            default_params["max_retry_count"] = args.max_retry_count
        if args.slide_batch_size is not None:
            default_params["slide_batch_size"] = args.slide_batch_size

        # Add model paths from pre-download or user-provided
        # Command-line args override config and pre-download
        
        # Set model_dir if provided
        if model_dir:
            default_params["model_dir"] = model_dir
            print(f"[Model Directory] Using: {model_dir}")

        # Parse models if provided
        models_list = None
        if args.models:
            models_list = [
                m.strip() for m in args.models.split(",")
            ]
        
        # Parse slide models if provided via config
        slide_models_list = None
        slide_model_types_str = config_defaults.get("slide_model_types")
        if slide_model_types_str:
            slide_models_list = [
                m.strip() for m in slide_model_types_str.split(",")
            ]

        # Remove parameters that are passed as explicit arguments to avoid conflicts
        task_default_params = default_params.copy()
        for key in [
            "container_image",
            "job_id",
            "pool_id",
            "output_prefix",
            "slides_per_task",
        ]:
            task_default_params.pop(key, None)

        # Use incremental staging and submission if Azure Blob staging is enabled
        if hasattr(args, 'stage_to_azure_blob') and args.stage_to_azure_blob:
            if args.stage_to_azure_blob and not args.staging_container:
                print(
                    "ERROR: --staging-container required when using --stage-to-azure-blob"
                )
                sys.exit(1)

            # For Azure Blob staging, stage slides and pass blob URLs to tasks
            if args.stage_to_azure_blob:
                print(f"\n[Azure Blob] Staging to container: {args.staging_container}")
                
                # Initialize blob staging
                if not submitter.azure_blob_staging:
                    if not args.storage_account_name or not args.storage_account_key:
                        print("ERROR: Azure storage account credentials required for blob staging")
                        sys.exit(1)
                    
                    submitter.azure_blob_staging = AzureBlobStaging(
                        account_name=args.storage_account_name,
                        account_key=args.storage_account_key,
                        container_name=args.staging_container,
                    )
                
                # Model staging removed - using persistent cache or model_dir
                # Models will be downloaded on-demand from Hugging Face with file locking to prevent clashes
                print(f"\n[Model Cache] Using persistent cache at /mnt/batch/tasks/cache")
                print(f"[Model Cache] Models will be downloaded on-demand from Hugging Face")
                
                # Read CSV and use incremental staging with parallel submission
                print(f"\n[Azure Blob] Incremental staging and submission enabled")
                import csv
                import subprocess
                import tempfile
                from concurrent.futures import ThreadPoolExecutor, as_completed
                from threading import Lock
                
                with open(args.csv_manifest, "r") as f:
                    reader = csv.DictReader(f)
                    slides = list(reader)
                
                print(f"[Azure Blob] Staging {len(slides)} slides with parallel workers...")
                
                # Batch-load all existing blobs for efficient checking
                print(f"[Azure Blob] Loading existing blob list for fast lookup...")
                existing_blobs = submitter.azure_blob_staging.get_blob_set()
                print(f"[Azure Blob] Found {len(existing_blobs)} existing blobs in storage")
                
                # Prepare default parameters for task submission
                task_default_params = default_params.copy()
                for key in [
                    "container_image",
                    "job_id",
                    "pool_id",
                    "output_prefix",
                    "slides_per_task",
                ]:
                    task_default_params.pop(key, None)
                
                # Parse models if provided
                models_list = None
                if args.models:
                    models_list = [m.strip() for m in args.models.split(",")]
                
                # Parse slide models if provided via config
                slide_models_list = None
                slide_model_types_str = config_defaults.get("slide_model_types")
                if slide_model_types_str:
                    slide_models_list = [m.strip() for m in slide_model_types_str.split(",")]
                
                # Ensure output container exists if remote output is specified
                if args.output_prefix:
                    submitter._ensure_output_container(args.output_prefix)
                
                # Shared state for incremental submission
                staged_slide_paths = {}
                staged_lock = Lock()
                batch_size = args.slides_per_task  # Submit tasks in batches
                submission_batch = []
                total_staged = 0
                total_submitted = 0
                total_batches_expected = (len(slides) + batch_size - 1) // batch_size  # Total batches across all slides
                current_batch_num = 0  # Track which batch we're on
                
                def stage_slide(slide_info):
                    """Stage a single slide (used by thread pool)."""
                    slide = slide_info
                    slide_id = slide.get("slide_id") or slide.get("sample_id") or slide.get("image_id")
                    slide_path = slide.get("slide_path") or slide.get("svs_path") or slide.get("path")
                    
                    if not slide_id or not slide_path:
                        return None, f"Could not find slide ID or path in CSV row: {slide}"
                    
                    slide_filename = os.path.basename(slide_path)
                    
                    # Check if already staged to blob (in slides/ or revision_slides/)
                    possible_blob_names = [
                        f"slides/{slide_filename}",
                        f"revision_slides/{slide_filename}",
                    ]
                    already_staged_name = None
                    for blob_name in possible_blob_names:
                        if blob_name in existing_blobs:
                            already_staged_name = blob_name
                            break
                    
                    if already_staged_name:
                        # File already staged to Blob - use it
                        blob_url = f"azblob://{args.storage_account_name}.blob.core.windows.net/{args.staging_container}/{already_staged_name}"
                        return (slide_id, blob_url), None
                    elif slide_path.startswith("azblob://"):
                        # Already an azblob:// URL - use it directly
                        return (slide_id, slide_path), None
                    elif slide_path.startswith("azfiles://"):
                        # Already an azfiles:// URL - use it directly
                        return (slide_id, slide_path), None
                    elif slide_path.startswith("s3://"):
                        # S3 file - download and upload to blob
                        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(slide_filename)[1], delete=False) as tmp:
                            tmp_path = tmp.name
                        try:
                            # Download from S3
                            result = subprocess.run(
                                ["aws", "s3", "cp", slide_path, tmp_path, "--no-progress"],
                                capture_output=True,
                                text=True,
                                env=os.environ.copy()
                            )
                            if result.returncode != 0:
                                return (slide_id, slide_path), f"Failed to download from S3: {result.stderr}"
                            else:
                                # Upload to blob
                                blob_name = f"slides/{slide_filename}"
                                submitter.azure_blob_staging.upload_file(
                                    local_path=tmp_path,
                                    blob_name=blob_name,
                                    show_progress=False,
                                )
                                blob_url = f"azblob://{args.storage_account_name}.blob.core.windows.net/{args.staging_container}/{blob_name}"
                                return (slide_id, blob_url), None
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                    elif slide_path.startswith(("http://", "https://")):
                        # HTTP URL - let the batch script handle it
                        return (slide_id, slide_path), None
                    else:
                        # Local path
                        if os.path.exists(slide_path):
                            blob_name = f"slides/{slide_filename}"
                            submitter.azure_blob_staging.upload_file(
                                local_path=slide_path,
                                blob_name=blob_name,
                                show_progress=False,
                            )
                            blob_url = f"azblob://{args.storage_account_name}.blob.core.windows.net/{args.staging_container}/{blob_name}"
                            return (slide_id, blob_url), None
                        else:
                            return (slide_id, slide_path), f"File not found: {slide_path}"
                
                def submit_batch_incrementally():
                    """Submit accumulated batch of slides as tasks."""
                    nonlocal total_submitted, submission_batch, current_batch_num
                    if not submission_batch:
                        return
                    
                    # Increment batch number
                    current_batch_num += 1
                    
                    # Check job state before attempting submission
                    try:
                        job = submitter.batch_client.job.get(args.job_id)
                        if job.state in ['completed', 'deleting', 'disabled']:
                            print(f"[Incremental Submit] WARNING: Job {args.job_id} is {job.state}, skipping batch submission")
                            submission_batch = []
                            return
                    except Exception as e:
                        print(f"[Incremental Submit] WARNING: Could not check job state: {e}")
                        # Continue anyway, let the submit attempt fail with proper error
                    
                    # Create temporary CSV for this batch
                    batch_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                    writer = csv.DictWriter(batch_csv, fieldnames=['slide_id', 'slide_path'])
                    writer.writeheader()
                    for slide_id in submission_batch:
                        writer.writerow({'slide_id': slide_id, 'slide_path': staged_slide_paths[slide_id]})
                    batch_csv.close()
                    
                    try:
                        # Submit this batch of tasks with global batch tracking
                        submitter.submit_tasks_from_csv(
                            job_id=args.job_id,
                            csv_file=batch_csv.name,
                            output_s3_prefix=args.output_prefix,
                            container_image=args.container_image,
                            models=models_list,
                            slide_models=slide_models_list,
                            slides_per_task=args.slides_per_task,
                            staged_slide_paths=staged_slide_paths,
                            use_container_prepull=args.use_container_prepull,
                            batch_offset=current_batch_num - 1,  # Zero-indexed offset
                            total_batches_global=total_batches_expected,
                            **task_default_params,
                        )
                        total_submitted += len(submission_batch)
                        print(f"[Incremental Submit] Submitted batch {current_batch_num}/{total_batches_expected}: {len(submission_batch)} slides ({total_submitted}/{len(slides)} total)")
                    except Exception as e:
                        if 'JobCompleted' in str(e) or 'already in a completed state' in str(e):
                            print(f"[Incremental Submit] Job {args.job_id} completed during staging - {len(submission_batch)} slides not submitted")
                        else:
                            print(f"[Incremental Submit] ERROR: Failed to submit batch: {e}")
                            raise
                    finally:
                        os.unlink(batch_csv.name)
                        submission_batch = []
                
                # Use ThreadPoolExecutor for parallel staging
                max_workers = min(args.staging_workers, len(slides))  # Up to configured parallel workers
                print(f"[Azure Blob] Using {max_workers} parallel workers for staging")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all staging tasks
                    future_to_slide = {executor.submit(stage_slide, slide): slide for slide in slides}
                    
                    # Process completed staging tasks as they finish
                    for future in as_completed(future_to_slide):
                        result, error = future.result()
                        if result:
                            slide_id, blob_url = result
                            with staged_lock:
                                staged_slide_paths[slide_id] = blob_url
                                submission_batch.append(slide_id)
                                total_staged += 1
                                
                                # Print progress
                                if error:
                                    print(f"  [{slide_id}] Staged with warning: {error}")
                                elif total_staged % 10 == 0:
                                    print(f"  Staged {total_staged}/{len(slides)} slides...")
                                
                                # Submit batch when we have enough slides
                                # Submit every batch_size slides to ensure steady task creation
                                if len(submission_batch) >= batch_size:
                                    submit_batch_incrementally()
                    
                    # Submit any remaining slides
                    with staged_lock:
                        if submission_batch:
                            submit_batch_incrementally()
                
                print(f"[Azure Blob] Incremental staging complete: {total_staged} slides staged, {total_submitted} slides submitted")
                
                # Check if some slides were not submitted due to job completion
                if total_staged > total_submitted:
                    unsubmitted = total_staged - total_submitted
                    print(f"\n[INFO] {unsubmitted} slides were staged but not submitted (job completed during staging)")
                    print(f"       These slides are available in blob storage for future jobs")
                
                # Cleanup staged slides if requested
                if hasattr(args, 'cleanup_staged_files') and args.cleanup_staged_files:
                    print(f"\n[Azure Blob] Cleaning up staged slides...")
                    for slide_id, blob_url in staged_slide_paths.items():
                        if blob_url.startswith("https://"):
                            blob_name = blob_url.split(f"{args.staging_container}/")[1]
                            try:
                                submitter.azure_blob_staging.delete_file(blob_name)
                            except Exception as e:
                                print(f"  Warning: Failed to delete {blob_name}: {e}")
                    print(f"[Azure Blob] Cleanup complete")
                
                return  # Exit early, blob staging workflow is complete
            
            # Azure Files staging workflow (existing code)
            # Model directory is staged in submit_tasks_from_csv if Azure Files is configured

            # Submit tasks using batch processing (not incremental staging)
            submitter.submit_tasks_from_csv(
                job_id=args.job_id,
                csv_file=args.csv_manifest,
                output_s3_prefix=args.output_prefix,
                container_image=args.container_image,
                models=models_list,
                slide_models=slide_models_list,
                use_container_prepull=args.use_container_prepull,
                **task_default_params,
            )

        else:
            # Standard workflow without staging
            submitter.submit_tasks_from_csv(
                job_id=args.job_id,
                csv_file=args.csv_manifest,
                output_s3_prefix=args.output_prefix,
                container_image=args.container_image,
                models=models_list,
                slide_models=slide_models_list,
                slides_per_task=args.slides_per_task,
                use_container_prepull=args.use_container_prepull,
                **task_default_params,
            )
    elif args.task_id and args.slide_path:
        # Single task submission
        # Pass model_dir if specified
        task_params = {}
        if model_dir:
            task_params["model_dir"] = model_dir

        submitter.submit_task(
            job_id=args.job_id,
            task_id=args.task_id,
            slide_path=args.slide_path,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            aws_region=args.aws_region,
            hf_token=args.hf_token,
            max_retry_count=args.max_retry_count,
            container_image=args.container_image,
            **task_params,
        )
    else:
        print(
            "ERROR: Must specify either --csv-manifest or --task-id with --slide-path"
        )
        sys.exit(1)

    # Monitor if requested
    if args.monitor:
        # If auto_delete_pool is enabled, delete the pool after all tasks complete
        submitter.monitor_tasks(
            job_id=args.job_id,
            delete_pool_on_completion=args.auto_delete_pool,
            pool_id=args.pool_id if args.auto_delete_pool else None
        )

    # Save failed tasks if requested
    if args.save_failed_tasks:
        submitter.save_failed_tasks(
            job_id=args.job_id,
            output_file=args.save_failed_tasks,
            task_metadata=submitter.task_metadata,
        )

    # Generate results manifest if requested
    if args.generate_manifest:
        submitter.generate_results_manifest(
            job_id=args.job_id,
            output_file=args.generate_manifest,
            task_metadata=submitter.task_metadata,
        )

    # Cleanup if requested
    if hasattr(args, 'cleanup_staged_files') and args.cleanup_staged_files:
        submitter.cleanup_staged_files()

    if args.delete_job:
        submitter.delete_job(job_id=args.job_id)

    # Handle pool deletion
    # Note: If auto_delete_pool is set with --monitor, pool is deleted in monitor_tasks()
    # If delete_pool is set without --monitor, delete immediately
    if args.delete_pool and not args.auto_delete_pool:
        submitter.delete_pool(pool_id=args.pool_id)
    elif args.auto_delete_pool and not args.monitor:
        print("\nWARNING: --auto-delete-pool requires --monitor to delete the pool after all tasks complete.")
        print("Use '--monitor --auto-delete-pool' together, or use '--delete-pool' to delete immediately.")


if __name__ == "__main__":
    main()
