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
        "WARNING: Could not import azure_files_staging module. Azure Files staging features will be unavailable."
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
        """Convert azblob://account/container/path to az://container/path format for fsspec."""
        if url and url.startswith("azblob://"):
            # azblob://account/container/path -> az://container/path
            parts = url.replace("azblob://", "").split("/", 1)
            if len(parts) >= 2:
                # Skip account name, keep container and path
                container_and_path = parts[1]
                return f"az://{container_and_path}"
            elif len(parts) == 1:
                # Just container, no path
                return f"az://{parts[0]}"
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
        # Formats: azblob://account/container/path or az://container/path
        container_name = None
        if output_prefix.startswith("azblob://"):
            # azblob://account.blob.core.windows.net/container/path
            parts = output_prefix.replace("azblob://", "").split("/")
            if len(parts) >= 2:
                container_name = parts[1]
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
        # Model pre-staging removed - using persistent cache instead
        # Models will be downloaded on-demand from Hugging Face to /mnt/batch_models
        model_download_cmd = ""
        
        if "ubuntu-hpc" in offer:
            # ubuntu-hpc images come with NVIDIA drivers and nvidia-docker2 pre-installed
            # Conditionally pull the image if not using pre-pull
            pull_image_cmd = "" if use_container_prepull else f"echo 'Pulling container image to /mnt/docker...'\n                docker pull {container_image}\n                "
            
            start_task_cmd = f'''/bin/bash -c "
                set -e
                echo 'Setting up Docker for GPU support...'
                
                # Move Docker data directory to temporary disk (has more space)
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
                # Restart Docker - may show error but will recover
                systemctl start docker || systemctl restart docker || true
                
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
                mkdir -p /mnt/batch_models
                chmod -R 777 /mnt/batch_models
                # Temporary working directories
                mkdir -p /mnt/batch/tasks/workitems/tmp
                chmod -R 777 /mnt/batch/tasks/workitems
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
                mkdir -p /mnt/batch_models
                chmod -R 777 /mnt/batch_models
                mkdir -p /mnt/batch/tasks/workitems/tmp
                chmod -R 777 /mnt/batch/tasks/workitems
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

        # Add Azure Files mount configuration if requested
        mount_config = None
        if mount_azure_files and self.azure_files_staging:
            print(
                f"  Configuring Azure Files mount: share '{self.azure_files_share_name}'"
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

    def create_job(self, job_id: str, pool_id: str) -> None:
        """Create a job in the specified pool."""
        print(f"Creating job '{job_id}'...")

        job = batchmodels.JobAddParameter(
            id=job_id,
            pool_info=batchmodels.PoolInformation(pool_id=pool_id),
        )

        try:
            self.batch_client.job.add(job)
            print(f"Job '{job_id}' created successfully")
        except batchmodels.BatchErrorException as e:
            if e.error.code == "JobExists":
                print(f"Job '{job_id}' already exists")
            else:
                raise

    def submit_task(
        self,
        job_id: str,
        task_id: str,
        slide_path: str = None,
        slide_paths: Optional[List[str]] = None,
        slide_ids: Optional[List[str]] = None,
        output_h5_path: str = None,
        output_pt_path: str = None,
        output_dir_for_batch: Optional[str] = None,
        intermediate_h5_path: Optional[str] = None,
        aggregation_method: str = "identity",
        slide_model_types: Optional[str] = None,
        slide_batch_size: int = 8,  # Batch size for slide encoding within a task
        classifier_pkl: Optional[str] = None,
        classifier_threshold: float = 0.75,
        prefilter_model_types: Optional[str] = None,
        prefilter_model_path: Optional[str] = None,
        model_types: Optional[str] = None,
        model_path: Optional[str] = None,
        slide_model_paths: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
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
        
        Args:
            script_blob_url: Optional Azure Blob URL to download scripts from (e.g., 
                https://account.blob.core.windows.net/container/scripts/).
                If provided, scripts will be downloaded at runtime instead of using bundled versions.
                This allows updating scripts without rebuilding the container.

        Supports both single-slide and multi-slide batch processing.
        For batch processing, provide slide_paths and slide_ids instead of slide_path.
        """
        print(f"Submitting task '{task_id}' to job '{job_id}'...")

        # Build environment variables
        env_vars = []

        # Handle batch vs single slide processing
        if slide_paths and len(slide_paths) > 1:
            # Batch processing mode
            env_vars.extend(
                [
                    batchmodels.EnvironmentSetting(
                        name="SLIDE_PATHS", value=",".join(slide_paths)
                    ),
                    batchmodels.EnvironmentSetting(
                        name="OUTPUT_DIR", value=output_dir_for_batch
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
        else:
            # Single slide mode (backward compatible)
            # If slide_paths has one element, use it; otherwise use slide_path parameter
            if slide_paths and not slide_path:
                slide_path = slide_paths[0]
            env_vars.extend(
                [
                    batchmodels.EnvironmentSetting(name="SLIDE_PATH", value=slide_path),
                    batchmodels.EnvironmentSetting(
                        name="OUTPUT_H5_PATH", value=output_h5_path
                    ),
                    batchmodels.EnvironmentSetting(
                        name="OUTPUT_PT_PATH", value=output_pt_path
                    ),
                ]
            )
            # Also set OUTPUT_DIR if remote output is requested (for upload support)
            if output_dir_for_batch:
                env_vars.append(
                    batchmodels.EnvironmentSetting(
                        name="OUTPUT_DIR", value=output_dir_for_batch
                    )
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
        
        # Only set prefilter model types if provided
        if prefilter_model_types:
            common_env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="PREFILTER_MODEL_TYPES", value=prefilter_model_types
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

        # Add model paths if provided
        if prefilter_model_path:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="PREFILTER_MODEL_PATH", value=prefilter_model_path
                )
            )
        if model_path:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="POSTFILTER_MODEL_PATH", value=model_path
                )
            )
        if slide_model_paths:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="SLIDE_MODEL_PATHS", value=slide_model_paths
                )
            )
        if model_cache_dir:
            env_vars.append(
                batchmodels.EnvironmentSetting(
                    name="MODEL_CACHE_DIR", value=model_cache_dir
                )
            )

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
        # - /mnt/batch/tasks/workitems: Azure Batch task working directory (local SSD, large space)
        # - /mnt/batch_models: Persistent model cache directory (survives across tasks)
        # - Use /mnt/batch/tasks/workitems for temp storage instead of /tmp (limited space)
        volume_mounts = (
            f"-v {azure_files_mount}:{azure_files_mount}" if azure_files_mount else ""
        )
        volume_mounts += " -v /mnt/batch/tasks/workitems:/mnt/batch/tasks/workitems"
        volume_mounts += " -v /mnt/batch/tasks/shared:/mnt/batch/tasks/shared"
        volume_mounts += " -v /mnt/batch_models:/mnt/batch_models"

        # Set cache and temp directories
        # Use persistent model cache at /mnt/batch_models (survives across tasks)
        # TMPDIR uses task working directory (has more space than /tmp)
        cache_env = "-e TORCH_HOME=/mnt/batch_models -e HF_HOME=/mnt/batch_models -e HF_HUB_CACHE=/mnt/batch_models/hub -e TRANSFORMERS_CACHE=/mnt/batch_models -e TMPDIR=/mnt/batch/tasks/workitems/tmp -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=1"

        task_command = f'/bin/bash -c "mkdir -p /mnt/batch_models /mnt/batch/tasks/workitems/tmp /mnt/batch/tasks/shared && chmod -R 777 /mnt/batch_models /mnt/batch/tasks/workitems && docker run --rm --user root --ipc host --gpus all --shm-size=8g {docker_env_args} {cache_env} {volume_mounts} {container_image} /bin/bash /app/scripts/azure_batch/run_tessellate_extract_features.sh"'

        # Task constraints with retry configuration
        task_constraints = batchmodels.TaskConstraints(
            max_task_retry_count=max_retry_count
        )

        # Configure automatic log upload to blob storage
        # This ensures logs are preserved even if the pool is deleted
        output_files = []
        if self.storage_account_name and self.storage_account_key:
            # Upload logs to a dedicated container
            log_container = "batch-logs"
            
            # Create SAS URL for log uploads (valid for 7 days)
            from datetime import datetime, timedelta
            from azure.storage.blob import generate_container_sas, ContainerSasPermissions
            
            sas_token = generate_container_sas(
                account_name=self.storage_account_name,
                container_name=log_container,
                account_key=self.storage_account_key,
                permission=ContainerSasPermissions(write=True, create=True, list=True),
                expiry=datetime.utcnow() + timedelta(days=7)
            )
            
            container_url = f"https://{self.storage_account_name}.blob.core.windows.net/{log_container}?{sas_token}"
            
            # Upload stdout
            output_files.append(
                batchmodels.OutputFile(
                    file_pattern='../stdout.txt',
                    destination=batchmodels.OutputFileDestination(
                        container=batchmodels.OutputFileBlobContainerDestination(
                            container_url=container_url,
                            path=f"{job_id}/{task_id}/stdout.txt"
                        )
                    ),
                    upload_options=batchmodels.OutputFileUploadOptions(
                        upload_condition=batchmodels.OutputFileUploadCondition.task_completion
                    )
                )
            )
            
            # Upload stderr
            output_files.append(
                batchmodels.OutputFile(
                    file_pattern='../stderr.txt',
                    destination=batchmodels.OutputFileDestination(
                        container=batchmodels.OutputFileBlobContainerDestination(
                            container_url=container_url,
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
            # Use the script from /app (copied during Docker build)
            # Create directories and run the batch script
            container_command = '-c "mkdir -p /mnt/batch/tasks/workitems/torch_cache /mnt/batch/tasks/workitems/hf_cache /mnt/batch/tasks/workitems/tmp /mnt/batch/tasks/shared && chmod -R 777 /mnt/batch/tasks/workitems && bash -x /app/scripts/azure_batch/run_tessellate_extract_features.sh"'
            
            # Build container run options with environment variables
            # Azure Batch environment_settings don't automatically pass into containers
            # We need to explicitly add them as Docker -e flags
            env_flags = []
            for env_var in env_vars:
                # For Docker -e flags, use single quotes to avoid shell interpretation
                # Replace single quotes in value with '\'' (end quote, escaped quote, start quote)
                value = env_var.value.replace("'", "'\\''") if env_var.value else ""
                env_flags.append(f"-e '{env_var.name}={value}'")
            
            container_run_options = '--rm --user=root --ipc=host --shm-size=8g --entrypoint=/bin/bash'
            if env_flags:
                container_run_options += ' ' + ' '.join(env_flags)
            
            # Azure Batch container settings
            # Note: GPU allocation is handled by Azure Batch at pool level, not container level
            # Override entrypoint to /bin/bash to avoid groupadd errors from default entrypoint
            # Use taskWorkingDirectory to access Azure Batch mounts (/mnt/batch/tasks/...)
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
                output_files=output_files if output_files else None,
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

        try:
            self.batch_client.task.add(job_id, task)

            # Store task metadata for failure tracking
            self.task_metadata[task_id] = {
                "slide_path": slide_path,
                "output_h5_path": output_h5_path,
                "output_pt_path": output_pt_path,
            }
            if intermediate_h5_path:
                self.task_metadata[task_id]["intermediate_h5_path"] = (
                    intermediate_h5_path
                )

            print(
                f"Task '{task_id}' submitted successfully (max retries: {max_retry_count})"
            )
        except batchmodels.BatchErrorException as e:
            print(f"Error submitting task: {e}")
            raise

    def stage_and_submit_tasks_from_csv(
        self,
        job_id: str,
        csv_file: str,
        output_s3_prefix: Optional[str] = None,
        container_image: str = "mskmind/mussel:latest-torch-gpu",
        models: Optional[List[str]] = None,
        slide_models: Optional[List[str]] = None,
        remote_dir: str = "slides",
        use_container_prepull: bool = False,
        **default_params,
    ) -> None:
        """
        Stage slides to Azure Files and submit tasks incrementally.

        This method stages each slide to Azure Files and immediately submits
        the corresponding task, allowing processing to start as soon as the
        first slide is staged rather than waiting for all slides to be staged.

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
            remote_dir: Remote directory for slides in Azure Files
            **default_params: Default parameters for all tasks (e.g., prefilter_model_type, batch_size)
        """
        if not self.azure_files_staging:
            raise ValueError(
                "Azure Files staging not configured. Provide storage account details and share name."
            )

        print(f"Loading task manifest from '{csv_file}'...")

        # Get prefilter model type (used for directory organization when single model)
        prefilter_model = default_params.get("prefilter_model_type", None)

        # Determine model type for directory structure
        if models and len(models) > 1:
            # Multiple models - use first model for base directory, but actual paths will be model-specific
            model_type = models[0]
            print(
                f"Will process each slide with {len(models)} models sequentially: {', '.join(models)}"
            )
        elif models and len(models) == 1:
            model_type = models[0]
        else:
            # Single model from default_params or prefilter
            model_type = default_params.get("model_type", prefilter_model)

        # Read CSV and process slides one by one
        tasks_submitted = 0
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            slides = list(reader)

        print(f"[Azure Files] Staging and submitting {len(slides)} slides...")
        print(f"[Azure Files] Tasks will start processing as slides are staged")

        for idx, row in enumerate(slides, 1):
            slide_id = row["slide_id"]
            slide_path = row["slide_path"]

            # Stage slide to Azure Files
            filename = os.path.basename(slide_path)
            remote_path = f"{remote_dir}/{filename}"

            print(f"[{idx}/{len(slides)}] Staging {slide_id}...")
            self.azure_files_staging.upload_file(slide_path, remote_path)

            # Create azfiles:// path
            azfiles_path = f"azfiles://{self.storage_account_name}/{self.azure_files_share_name}/{remote_path}"

            # Track for cleanup
            self.staged_files.append(remote_path)

            # Create task with staged path
            task_id = slide_id

            # For multi-model, base output paths use the first model's directory
            # The bash script will handle creating model-specific subdirectories
            base_prefix = output_s3_prefix.rstrip("/") if output_s3_prefix else "/mnt/batch/tasks/shared/output"
            output_h5_path = f"{base_prefix}/{model_type}/h5/{slide_id}_features.h5"
            output_pt_path = f"{base_prefix}/{model_type}/pt/{slide_id}_features.pt"
            # Only set intermediate_h5_path if aggregation method requires it
            if self._should_set_intermediate_h5_path(
                default_params.get("aggregation_method")
            ):
                intermediate_h5_path = f"{base_prefix}/{model_type}/tile_h5/{slide_id}_tile_features.h5"
            else:
                intermediate_h5_path = None

            # Merge with default parameters
            merged_config = {**default_params}
            merged_config["task_id"] = task_id
            merged_config["slide_path"] = azfiles_path
            merged_config["output_h5_path"] = output_h5_path
            merged_config["output_pt_path"] = output_pt_path
            if intermediate_h5_path:
                merged_config["intermediate_h5_path"] = intermediate_h5_path

            # Add model types as comma-separated list
            if models:
                merged_config["model_types"] = ",".join(models)
            
            # Add slide models as comma-separated list
            if slide_models:
                merged_config["slide_model_types"] = ",".join(slide_models)

            # Submit task immediately
            print(f"[{idx}/{len(slides)}] Submitting task for {slide_id}...")
            self.submit_task(
                job_id=job_id,
                task_id=merged_config["task_id"],
                slide_path=merged_config["slide_path"],
                output_h5_path=merged_config["output_h5_path"],
                output_pt_path=merged_config["output_pt_path"],
                intermediate_h5_path=merged_config.get("intermediate_h5_path"),
                aggregation_method=merged_config.get("aggregation_method", "identity"),
                classifier_pkl=merged_config.get("classifier_pkl"),
                classifier_threshold=merged_config.get("classifier_threshold", 0.75),
                prefilter_model_types=merged_config.get(
                    "prefilter_model_types", None
                ),
                prefilter_model_path=merged_config.get("prefilter_model_path"),
                model_types=merged_config.get("model_types"),
                model_path=merged_config.get("model_path"),
                slide_model_types=merged_config.get("slide_model_types"),
                slide_model_paths=merged_config.get("slide_model_paths"),
                model_cache_dir=merged_config.get("model_cache_dir"),
                seg_config_group=merged_config.get("seg_config_group"),
                segment_threshold=merged_config.get("segment_threshold"),
                patch_size=merged_config.get("patch_size"),
                step_size=merged_config.get("step_size"),
                mpp=merged_config.get("mpp"),
                seg_level=merged_config.get("seg_level"),
                segment_max_value=merged_config.get("segment_max_value"),
                median_blur_ksize=merged_config.get("median_blur_ksize"),
                morphology_ex_kernel=merged_config.get("morphology_ex_kernel"),
                ref_patch_size=merged_config.get("ref_patch_size"),
                use_otsu=merged_config.get("use_otsu"),
                tissue_area_threshold=merged_config.get("tissue_area_threshold"),
                hole_area_threshold=merged_config.get("hole_area_threshold"),
                max_num_holes=merged_config.get("max_num_holes"),
                num_workers=merged_config.get("num_workers", 4),
                batch_size=merged_config.get("batch_size", 64),
                use_gpu=merged_config.get("use_gpu", True),
                keep_intermediate_files=merged_config.get(
                    "keep_intermediate_files", False
                ),
                hf_token=merged_config.get("hf_token"),
                aws_access_key_id=merged_config.get("aws_access_key_id"),
                aws_secret_access_key=merged_config.get("aws_secret_access_key"),
                aws_region=merged_config.get("aws_region"),
                max_retry_count=merged_config.get("max_retry_count", 3),
                container_image=container_image,
                cleanup_staged_file=True,  # Enable per-task cleanup for staged files
                use_container_prepull=use_container_prepull,
            )
            tasks_submitted += 1

        print(f"\n[Azure Files] Staged and submitted {tasks_submitted} tasks")

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
        prefilter_models_str = default_params.get("prefilter_model_types", None)
        prefilter_models = [m.strip() for m in prefilter_models_str.split(",")] if prefilter_models_str else []
        prefilter_model = prefilter_models[0] if prefilter_models else None

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

        # Read all slides from CSV
        slides = []
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
                    print(f"  Using staged path for {slide_id}: {slide_path}")

                slides.append({"slide_id": slide_id, "slide_path": slide_path})

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
        
            # Stage model files to Azure Files if enabled
            if self.azure_files_staging:
                # Stage classifier model
                classifier_pkl = default_params.get("classifier_pkl")
                if classifier_pkl and not classifier_pkl.startswith(
                    ("s3://", "azfiles://", "http://", "https://")
                ):
                    if os.path.exists(classifier_pkl):
                        self.log(
                            f"Staging classifier model: {os.path.basename(classifier_pkl)}"
                        )
                        remote_path = f"models/{os.path.basename(classifier_pkl)}"
                        self.azure_files_staging.upload_file(
                            local_path=classifier_pkl,
                            remote_path=remote_path,
                            show_progress=False,
                        )
                        azfiles_url = f"azfiles://{self.storage_account_name}/{self.azure_files_share_name}/{remote_path}"
                        default_params["classifier_pkl"] = azfiles_url
                        self.log(f"  Staged to: {remote_path}")

                # Stage prefilter model
                prefilter_model_path = default_params.get("prefilter_model_path")
                if prefilter_model_path and not prefilter_model_path.startswith(
                    ("s3://", "azfiles://", "http://", "https://")
                ):
                    if os.path.exists(prefilter_model_path):
                        self.log(
                            f"Staging prefilter model: {os.path.basename(prefilter_model_path)}"
                        )
                        remote_path = f"models/{os.path.basename(prefilter_model_path)}"
                        self.azure_files_staging.upload_file(
                            local_path=prefilter_model_path,
                            remote_path=remote_path,
                            show_progress=False,
                        )
                        azfiles_url = f"azfiles://{self.storage_account_name}/{self.azure_files_share_name}/{remote_path}"
                        default_params["prefilter_model_path"] = azfiles_url
                        self.log(f"  Staged to: {remote_path}")

                # Stage model
                model_path = default_params.get("model_path")
                if model_path and not model_path.startswith(
                    ("s3://", "azfiles://", "http://", "https://")
                ):
                    if os.path.exists(model_path):
                        self.log(
                            f"Staging model: {os.path.basename(model_path)}"
                        )
                        remote_path = (
                            f"models/{os.path.basename(model_path)}"
                        )
                        self.azure_files_staging.upload_file(
                            local_path=model_path,
                            remote_path=remote_path,
                            show_progress=False,
                        )
                        azfiles_url = f"azfiles://{self.storage_account_name}/{self.azure_files_share_name}/{remote_path}"
                        default_params["model_path"] = azfiles_url
                        self.log(f"  Staged to: {remote_path}")

            # Group slides into batches
            for batch_idx in range(0, len(slides), slides_per_task):
                    batch_slides = slides[
                        batch_idx : batch_idx + slides_per_task
                    ]

                    # Create batch task ID
                    batch_num = batch_idx // slides_per_task + 1
                    total_batches = (len(slides) + slides_per_task - 1) // slides_per_task
                    
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

                    # Stage local files to Azure Files or Azure Blob if enabled
                    if self.azure_files_staging or self.azure_blob_staging:
                        staged_paths = []
                        for slide_path in slide_paths_batch:
                            # Check if it's a local file path (not s3://, azfiles://, azblob://, or http/https)
                            if not slide_path.startswith(
                                ("s3://", "azfiles://", "azblob://", "http://", "https://")
                            ):
                                if os.path.exists(slide_path):
                                    slide_filename = os.path.basename(slide_path)
                            
                                    # Stage to Azure Blob if available
                                    if self.azure_blob_staging:
                                        self.log(f"Staging to Azure Blob: {slide_filename}")
                                        blob_name = f"slides/{slide_filename}"
                                        self.azure_blob_staging.upload_file(
                                            local_path=slide_path,
                                            blob_name=blob_name,
                                            show_progress=False,
                                        )
                                        # Convert to azblob:// URL
                                        azblob_url = f"azblob://{self.storage_account_name}/{self.staging_container}/{blob_name}"
                                        staged_paths.append(azblob_url)
                                    # Otherwise stage to Azure Files if available
                                    elif self.azure_files_staging:
                                        self.log(f"Staging to Azure Files: {slide_filename}")
                                        remote_path = f"slides/{slide_filename}"
                                        self.azure_files_staging.upload_file(
                                            local_path=slide_path,
                                            remote_path=remote_path,
                                            show_progress=False,
                                        )
                                        # Convert to azfiles:// URL
                                        azfiles_url = f"azfiles://{self.storage_account_name}/{self.azure_files_share_name}/{remote_path}"
                                        staged_paths.append(azfiles_url)
                                else:
                                    self.log(f"WARNING: Local file not found: {slide_path}")
                                    staged_paths.append(slide_path)
                            elif slide_path.startswith("s3://"):
                                # Check if S3 file is already staged to Azure Blob or Azure Files
                                slide_filename = os.path.basename(slide_path)
                        
                                # Check multiple possible remote directories for both Azure Blob and Files
                                possible_blob_names = [
                                    f"slides/{slide_filename}",
                                    f"revision_slides/{slide_filename}",
                                ]
                        
                                # First check Azure Blob if available
                                staged_blob_name = None
                                if self.azure_blob_staging:
                                    for blob_name in possible_blob_names:
                                        if self.azure_blob_staging.blob_exists(blob_name):
                                            staged_blob_name = blob_name
                                            break
                        
                                if staged_blob_name:
                                    # File already staged to Blob - use azblob:// URL
                                    azblob_url = f"azblob://{self.storage_account_name}/{self.staging_container}/{staged_blob_name}"
                                    self.log(f"Using already-staged slide (Blob): {slide_filename} (from {staged_blob_name})")
                                    staged_paths.append(azblob_url)
                                else:
                                    # Check Azure Files if available
                                    staged_remote_path = None
                                    if self.azure_files_staging:
                                        for remote_path in possible_blob_names:  # Use same paths
                                            if self.azure_files_staging.file_exists(remote_path):
                                                staged_remote_path = remote_path
                                                break
                            
                                    if staged_remote_path:
                                        # File already staged to Files - use azfiles:// URL
                                        azfiles_url = f"azfiles://{self.storage_account_name}/{self.azure_files_share_name}/{staged_remote_path}"
                                        self.log(f"Using already-staged slide (Files): {slide_filename} (from {staged_remote_path})")
                                        staged_paths.append(azfiles_url)
                                    else:
                                        # Not staged yet - keep S3 path (will be downloaded by container)
                                        staged_paths.append(slide_path)
                            else:
                                # Already a remote path (azfiles://, azblob://, or http://)
                                staged_paths.append(slide_path)
                        slide_paths_batch = staged_paths

                    print(f"\nSubmitting batch task: {batch_id}")
                    print(f"  Models: {', '.join(models_to_process)}")
                    print(f"  Slides: {', '.join(slide_ids)}")

                    # Use output prefix as-is (CLI will add model subdirectory)
                    if output_s3_prefix:
                        # Convert azblob:// to az:// for fsspec compatibility
                        output_dir_for_batch = self.convert_azblob_to_fsspec_url(output_s3_prefix.rstrip('/'))
                    else:
                        output_dir_for_batch = "/mnt/batch/tasks/shared/output"

                    # Submit batch task
                    self.submit_task(
                    job_id=job_id,
                    task_id=batch_id,
                    slide_paths=slide_paths_batch,
                    slide_ids=slide_ids,
                    output_dir_for_batch=output_dir_for_batch,
                    slide_batch_size=model_params.get("slide_batch_size", 8),
                    aggregation_method=model_params.get(
                        "aggregation_method", "identity"
                    ),
                    classifier_pkl=model_params.get("classifier_pkl"),
                    classifier_threshold=model_params.get(
                        "classifier_threshold", 0.75
                    ),
                    prefilter_model_types=model_params.get(
                        "prefilter_model_types", None
                    ),
                    prefilter_model_path=model_params.get("prefilter_model_path"),
                    model_types=model_params.get("model_types"),
                    model_path=model_params.get("model_path"),
                    slide_model_types=model_params.get("slide_model_types"),
                    slide_model_paths=model_params.get("slide_model_paths"),
                    model_cache_dir=model_params.get("model_cache_dir"),
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

                    # Store task configuration in metadata (excluding secrets)
                    if add_config_to_metadata:
                        add_config_to_metadata(self.task_metadata, model_params, batch_id)
                
                    total_tasks_submitted += 1

                # End of batch loop
            
        print(f"\n{'='*80}")
        print(f"Submitted {total_tasks_submitted} total tasks")
        print(f"  Models per task: {len(models_to_process)}")
        print(f"  Slides per task: {slides_per_task}")
        print(f"{'='*80}")
        return

    def monitor_tasks(self, job_id: str, poll_interval: int = 30) -> None:
        """Monitor task progress."""
        print(f"Monitoring tasks in job '{job_id}'...")
        print("Press Ctrl+C to stop monitoring (tasks will continue running)")

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
            elif task.state in [batchmodels.TaskState.failed]:
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
                task_info["output_h5_path"] = metadata.get("output_h5_path", "")
                task_info["output_pt_path"] = metadata.get("output_pt_path", "")

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

                # Extract model type from output path
                output_h5 = metadata.get("output_h5_path", "")
                if "/" in output_h5:
                    parts = output_h5.split("/")
                    # Look for model type in path (e.g., /CTRANSPATH/h5/)
                    for i, part in enumerate(parts):
                        if i < len(parts) - 1 and parts[i + 1] in [
                            "h5",
                            "pt",
                            "tile_h5",
                        ]:
                            task_info["model_type"] = part
                            break

                # Extract file type from path
                if output_h5.endswith(".h5"):
                    if "tile" in output_h5:
                        task_info["file_type"] = "tile_h5"
                    else:
                        task_info["file_type"] = "h5"
            else:
                # Extract from environment variables if available
                if task.environment_settings:
                    for env_var in task.environment_settings:
                        if env_var.name == "SLIDE_PATH":
                            task_info["slide_path"] = env_var.value
                        elif env_var.name == "OUTPUT_H5_PATH":
                            task_info["output_h5_path"] = env_var.value
                        elif env_var.name == "OUTPUT_PT_PATH":
                            task_info["output_pt_path"] = env_var.value
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

    def stage_slides_to_azure_files(
        self,
        slides: List[Dict[str, str]],
        remote_dir: str = "slides",
    ) -> Dict[str, str]:
        """
        Stage slides to Azure Files before processing.

        Args:
            slides: List of dicts with 'slide_id' and 'slide_path' keys
            remote_dir: Remote directory for slides in Azure Files

        Returns:
            Dict mapping slide_id to Azure Files path (azfiles://...)
        """
        if not self.azure_files_staging:
            raise ValueError(
                "Azure Files staging not configured. Provide storage account details and share name."
            )

        staged_paths = {}
        print(
            f"[Azure Files] Staging {len(slides)} slides to Azure Files share '{self.azure_files_share_name}'..."
        )

        for slide in slides:
            # Handle different column names flexibly
            slide_id = slide.get("slide_id") or slide.get("sample_id") or slide.get("image_id")
            slide_path = slide.get("slide_path") or slide.get("svs_path") or slide.get("path")
            
            if not slide_id or not slide_path:
                raise ValueError(f"Could not find slide ID or path in CSV row: {slide}")

            # Determine filename from path
            filename = os.path.basename(slide_path)

            # Upload to Azure Files
            remote_path = f"{remote_dir}/{filename}"
            self.azure_files_staging.upload_file(slide_path, remote_path)

            # Store mapping with azfiles:// URL format
            azfiles_path = f"azfiles://{self.storage_account_name}/{self.azure_files_share_name}/{remote_path}"
            staged_paths[slide_id] = azfiles_path

            # Track for cleanup
            self.staged_files.append(remote_path)

        print(f"[Azure Files] Staged {len(staged_paths)} slides")
        return staged_paths

    def cleanup_staged_files(self) -> None:
        """
        Clean up all staged files from Azure Files.

        Note: When using incremental staging (stage_and_submit_tasks_from_csv),
        files are automatically cleaned up after each task completes. This method
        is primarily for cleaning up files staged using stage_slides_to_azure_files
        or for manual cleanup operations.
        """
        if not self.azure_files_staging or not self.staged_files:
            return

        print(f"[Azure Files] Cleaning up {len(self.staged_files)} staged files...")

        # Group files by directory for efficient cleanup
        directories = set()
        for file_path in self.staged_files:
            directory = os.path.dirname(file_path)
            if directory:
                directories.add(directory)

        # Delete directories (which will delete all files within)
        if directories:
            self.azure_files_staging.cleanup_staging(list(directories))

        self.staged_files = []
        print("[Azure Files] Cleanup complete")

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

    # Azure Files staging configuration
    parser.add_argument(
        "--azure-files-share-name", help="Azure Files share name for staging files"
    )
    parser.add_argument(
        "--stage-to-azure-files",
        action="store_true",
        help="Stage input files to Azure Files before processing",
    )
    parser.add_argument(
        "--mount-azure-files",
        action="store_true",
        help="Mount Azure Files share to batch pool nodes",
    )
    parser.add_argument(
        "--cleanup-staged-files",
        action="store_true",
        help="Clean up staged files from Azure Files after processing",
    )

    # Azure Blob staging configuration
    parser.add_argument(
        "--staging-container",
        help="Azure Blob container name for staging slides and models (default: uses output_prefix for models)",
    )
    parser.add_argument(
        "--azure-blob-container",
        dest="staging_container",
        help="(Deprecated) Use --staging-container instead",
    )
    parser.add_argument(
        "--stage-to-azure-blob",
        action="store_true",
        help="Stage input files to Azure Blob Storage before processing",
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
        "Note: Not applicable when using --stage-to-azure-files (incremental staging). "
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
        "--model-cache-dir",
        default="./model_cache",
        help="Local directory to cache models (default: ./model_cache)",
    )
    parser.add_argument(
        "--model-s3-prefix",
        help="S3 prefix for uploaded models (default: use output-s3-prefix/models/)",
    )
    parser.add_argument(
        "--model-dir",
        help="Path to directory containing pre-downloaded models (azblob://, s3://, or local path). When specified, disables automatic model download and uses this directory instead.",
    )
    parser.add_argument(
        "--prefilter-model-path",
        help="Path to prefilter model weights (local or s3://)",
    )
    parser.add_argument(
        "--model-path",
        help="Path to model weights (local or s3://)",
    )
    parser.add_argument(
        "--slide-model-path",
        help="Path to slide encoder model weights (local or s3://)",
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

        if not args.mount_azure_files and config_defaults.get("mount_azure_files"):
            args.mount_azure_files = config_defaults["mount_azure_files"]

        if not args.stage_to_azure_files and config_defaults.get(
            "stage_to_azure_files"
        ):
            args.stage_to_azure_files = config_defaults["stage_to_azure_files"]

        if not args.stage_to_azure_blob and config_defaults.get(
            "stage_to_azure_blob"
        ):
            args.stage_to_azure_blob = config_defaults["stage_to_azure_blob"]

        if not args.cleanup_staged_files and config_defaults.get(
            "cleanup_staged_files"
        ):
            args.cleanup_staged_files = config_defaults["cleanup_staged_files"]

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

    # Pre-download models removed - using persistent cache instead
    # Models will be downloaded on-demand from Hugging Face to /mnt/batch_models
    model_paths = {}
    model_dir = None

    # Apply user-provided model paths if specified
    if args.prefilter_model_path:
        # Use the actual prefilter model type
        prefilter_model_type = getattr(
            args, "prefilter_model_type", None
        ) or config_defaults.get("prefilter_model_type", None)
        if prefilter_model_type:
            model_paths[prefilter_model_type] = args.prefilter_model_path
    if args.model_path:
        # Apply to all models if multiple
        if args.models:
            for model in args.models.split(","):
                model_paths[model.strip()] = args.model_path
    if args.slide_model_path:
        # Store slide model path with actual model type name
        slide_model_type = getattr(
            args, "slide_model_type", None
        ) or config_defaults.get("slide_model_type")
        if slide_model_type:
            model_paths[slide_model_type] = args.slide_model_path
        else:
            # Fallback to generic "slide" key if no type specified
            model_paths["slide"] = args.slide_model_path

    # Validate CTRANSPATH configuration
    # CTRANSPATH requires a model_path to be provided via configuration
    if args.csv_manifest or args.config_file:
        # Determine the prefilter model type from config or default
        # config_defaults is already loaded above
        prefilter_model = config_defaults.get("prefilter_model_type", None)

        # Check if CTRANSPATH is being used without a model_path
        # Check command-line, config file, and pre-downloaded paths
        has_prefilter_path = (
            args.prefilter_model_path
            or config_defaults.get("prefilter_model_path")
            or model_paths.get("CTRANSPATH")
        )

        if prefilter_model and prefilter_model.upper() == "CTRANSPATH" and not has_prefilter_path:
            print(
                "\n⚠️  WARNING: CTRANSPATH model requires a model_path to be provided via configuration"
            )
            print(
                "   CTRANSPATH does not have a default HuggingFace path and cannot be automatically downloaded."
            )
            print(
                "   Please provide the model path using one of the following methods:"
            )
            print(
                "     1. Command line: --prefilter-model-path /path/to/ctranspath.pth"
            )
            print(
                "     2. Configuration file: prefilter_model_path: /path/to/ctranspath.pth"
            )
            print("   Tasks will fail if CTRANSPATH model path is not provided.\n")

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

    # Model staging removed - using persistent cache at /mnt/batch_models instead
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
        submitter.create_job(job_id=args.job_id, pool_id=args.pool_id)

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
        
        # Use persistent model cache directory - models downloaded on-demand from Hugging Face
        # The persistent cache at /mnt/batch_models survives across tasks and reduces redundant downloads
        default_params["model_cache_dir"] = "/mnt/batch_models"
        print(f"[Model Cache] Using persistent cache at: /mnt/batch_models")
        print(f"[Model Cache] Models will be downloaded on-demand from Hugging Face")
        
        # Individual model paths can still override the cache if provided
        if model_paths.get("CTRANSPATH"):
            default_params["prefilter_model_path"] = model_paths["CTRANSPATH"]
            if args.prefilter_model_path:
                default_params["prefilter_model_path"] = args.prefilter_model_path
            # For postfilter, we'll pass the path that applies to all models
            # The model_path will be used for all models in the list
            if args.models and model_paths:
                # Use the first model's path if available
                first_model = args.models.split(",")[0].strip()
                if first_model in model_paths:
                    default_params["model_path"] = model_paths[first_model]
            if args.model_path:
                default_params["model_path"] = args.model_path
            if model_paths.get("slide"):
                default_params["slide_model_path"] = model_paths["slide"]
            if args.slide_model_path:
                default_params["slide_model_path"] = args.slide_model_path

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

        # Use incremental staging and submission if Azure Files or Blob staging is enabled
        if args.stage_to_azure_files or args.stage_to_azure_blob:
            if args.stage_to_azure_files and not args.azure_files_share_name:
                print(
                    "ERROR: --azure-files-share-name required when using --stage-to-azure-files"
                )
                sys.exit(1)
            
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
                
                # Apply pre-downloaded model paths to task_default_params if not already set
                if model_paths:
                    # Apply prefilter model path
                    prefilter_model_type = task_default_params.get("prefilter_model_type")
                    if prefilter_model_type and prefilter_model_type in model_paths:
                        if not task_default_params.get("prefilter_model_path"):
                            task_default_params["prefilter_model_path"] = model_paths[prefilter_model_type]
                            print(f"[Pre-download] Applied {prefilter_model_type} path: {model_paths[prefilter_model_type]}")
                    
                    # Apply model path
                    model_type = task_default_params.get("model_type")
                    if model_type and model_type in model_paths:
                        if not task_default_params.get("model_path"):
                            task_default_params["model_path"] = model_paths[model_type]
                            print(f"[Pre-download] Applied {model_type} path: {model_paths[model_type]}")
                    
                    # Apply slide model path
                    slide_model_type = task_default_params.get("slide_model_type")
                    if slide_model_type and slide_model_type in model_paths:
                        if not task_default_params.get("slide_model_path"):
                            task_default_params["slide_model_path"] = model_paths[slide_model_type]
                            print(f"[Pre-download] Applied {slide_model_type} path: {model_paths[slide_model_type]}")
                
                # Model staging removed - using persistent cache at /mnt/batch_models instead
                # Models will be downloaded on-demand from Hugging Face with file locking to prevent clashes
                print(f"\n[Model Cache] Using persistent cache at /mnt/batch_models")
                print(f"[Model Cache] Models will be downloaded on-demand from Hugging Face")
                
                # Read CSV and stage slides to blob
                print(f"\n[Azure Blob] Staging slides...")
                import csv
                import subprocess
                import tempfile
                staged_slide_paths = {}
                with open(args.csv_manifest, "r") as f:
                    reader = csv.DictReader(f)
                    slides = list(reader)
                
                print(f"[Azure Blob] Staging {len(slides)} slides...")
                for slide in slides:
                    # Handle different column names flexibly
                    slide_id = slide.get("slide_id") or slide.get("sample_id") or slide.get("image_id")
                    slide_path = slide.get("slide_path") or slide.get("svs_path") or slide.get("path")
                    
                    if not slide_id or not slide_path:
                        raise ValueError(f"Could not find slide ID or path in CSV row: {slide}")
                    
                    slide_filename = os.path.basename(slide_path)
                    
                    # Check if already staged to blob (in slides/ or revision_slides/)
                    possible_blob_names = [
                        f"slides/{slide_filename}",
                        f"revision_slides/{slide_filename}",
                    ]
                    already_staged_name = None
                    for blob_name in possible_blob_names:
                        if submitter.azure_blob_staging.blob_exists(blob_name):
                            already_staged_name = blob_name
                            break
                    
                    if already_staged_name:
                        # File already staged to Blob - use it
                        blob_url = f"azblob://{args.storage_account_name}.blob.core.windows.net/{args.staging_container}/{already_staged_name}"
                        print(f"  [{slide_id}] Using already-staged: {slide_filename} (from {already_staged_name})")
                        staged_slide_paths[slide_id] = blob_url
                    elif slide_path.startswith("azblob://"):
                        # Already an azblob:// URL - use it directly
                        staged_slide_paths[slide_id] = slide_path
                    elif slide_path.startswith("azfiles://"):
                        # Already an azfiles:// URL - use it directly
                        staged_slide_paths[slide_id] = slide_path
                    elif slide_path.startswith("s3://"):
                        # S3 file - download and upload to blob
                        print(f"  [{slide_id}] Staging from S3: {slide_filename}...")
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
                                print(f"  [{slide_id}] WARNING: Failed to download from S3: {result.stderr}")
                                staged_slide_paths[slide_id] = slide_path  # Fall back to S3 path
                            else:
                                # Upload to blob
                                blob_name = f"slides/{slide_filename}"
                                submitter.azure_blob_staging.upload_file(
                                    local_path=tmp_path,
                                    blob_name=blob_name,
                                    show_progress=False,
                                )
                                blob_url = f"azblob://{args.storage_account_name}.blob.core.windows.net/{args.staging_container}/{blob_name}"
                                staged_slide_paths[slide_id] = blob_url
                                print(f"  [{slide_id}] Staged to blob: {blob_name}")
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                    elif slide_path.startswith(("http://", "https://")):
                        # HTTP URL - let the batch script handle it
                        staged_slide_paths[slide_id] = slide_path
                    else:
                        # Local path
                        if os.path.exists(slide_path):
                            blob_name = f"slides/{slide_filename}"
                            print(f"  [{slide_id}] Staging {slide_filename}...")
                            submitter.azure_blob_staging.upload_file(
                                local_path=slide_path,
                                blob_name=blob_name,
                                show_progress=False,
                            )
                            blob_url = f"azblob://{args.storage_account_name}.blob.core.windows.net/{args.staging_container}/{blob_name}"
                            staged_slide_paths[slide_id] = blob_url
                        else:
                            print(f"  [{slide_id}] WARNING: File not found: {slide_path}")
                            staged_slide_paths[slide_id] = slide_path
                
                print(f"[Azure Blob] Staged {len(staged_slide_paths)} slides")
                
                # Ensure output container exists if remote output is specified
                if args.output_prefix:
                    submitter._ensure_output_container(args.output_prefix)
                
                # Now submit tasks with blob URLs (no incremental staging)
                submitter.submit_tasks_from_csv(
                    job_id=args.job_id,
                    csv_file=args.csv_manifest,
                    output_s3_prefix=args.output_prefix,
                    container_image=args.container_image,
                    models=models_list,
                    slide_models=slide_models_list,
                    slides_per_task=args.slides_per_task,
                    staged_slide_paths=staged_slide_paths,
                    use_container_prepull=args.use_container_prepull,
                    **task_default_params,
                )
                
                # Cleanup staged slides if requested
                if args.cleanup_staged_files:
                    print(f"\n[Azure Blob] Cleaning up staged slides...")
                    for slide_id, blob_url in staged_slide_paths.items():
                        if blob_url.startswith("https://"):
                            blob_name = blob_url.split(f"{args.staging_container}/")[1]
                            try:
                                submitter.azure_blob_staging.delete_file(blob_name)
                                print(f"  Deleted: {blob_name}")
                            except Exception as e:
                                print(f"  Warning: Failed to delete {blob_name}: {e}")
                    print(f"[Azure Blob] Cleanup complete")
                
                return  # Exit early, blob staging workflow is complete
            
            # Azure Files staging workflow (existing code)
            if args.stage_to_azure_files:
                # Stage models to Azure Files if we have any
                if (
                    model_paths
                    and stage_models_to_azure_files
                    and args.storage_account_name
                    and args.storage_account_key
                    and args.azure_files_share_name  # Only stage if Azure Files is configured
                ):
                    print("\n[Model Staging] Staging models to Azure Files...")
                    staged_model_paths = stage_models_to_azure_files(
                        model_paths=model_paths,
                        account_name=args.storage_account_name,
                        account_key=args.storage_account_key,
                        share_name=args.azure_files_share_name,
                        remote_dir="models",
                    )
                    # Update model paths to Azure Files paths (will be accessible at /mnt/batch/tasks/fsmounts/azfiles/models/)
                    for model_type, remote_path in staged_model_paths.items():
                        azfiles_model_path = (
                            f"/mnt/batch/tasks/fsmounts/azfiles/{remote_path}"
                        )
                        print(f"[Model Staging] Staged {model_type}: {azfiles_model_path}")
                        if (
                            model_type == "CTRANSPATH"
                            or task_default_params.get("prefilter_model_type") == model_type
                        ):
                            task_default_params["prefilter_model_path"] = azfiles_model_path
                            print(
                                f"[Model Staging] Set prefilter_model_path: {azfiles_model_path}"
                            )
                        elif model_type == task_default_params.get("model_type"):
                            task_default_params["model_path"] = (
                                azfiles_model_path
                            )
                            print(
                                f"[Model Staging] Set model_path: {azfiles_model_path}"
                            )
                        elif model_type == task_default_params.get("slide_model_type"):
                            task_default_params["slide_model_path"] = azfiles_model_path
                            print(
                                f"[Model Staging] Set slide_model_path: {azfiles_model_path}"
                            )
                    print(f"[Model Staging] Models staged and paths updated")

                # Also stage local model paths from config if not already staged
                if submitter.azure_files_staging:
                    # Stage prefilter model if it's a local path
                    prefilter_model_path = task_default_params.get("prefilter_model_path")
                    if prefilter_model_path and not prefilter_model_path.startswith(
                        ("s3://", "azfiles://", "http://", "https://", "/mnt/batch")
                    ):
                        if os.path.exists(prefilter_model_path):
                            print(
                                f"[Model Staging] Staging local prefilter model: {os.path.basename(prefilter_model_path)}"
                            )
                            remote_path = f"models/{os.path.basename(prefilter_model_path)}"
                            submitter.azure_files_staging.upload_file(
                                local_path=prefilter_model_path,
                                remote_path=remote_path,
                                show_progress=False,
                            )
                            azfiles_model_path = (
                                f"/mnt/batch/tasks/fsmounts/azfiles/{remote_path}"
                            )
                            task_default_params["prefilter_model_path"] = azfiles_model_path
                            print(
                                f"[Model Staging] Updated prefilter_model_path: {azfiles_model_path}"
                            )

                    # Stage model if it's a local path
                    model_path = task_default_params.get("model_path")
                    if model_path and not model_path.startswith(
                        ("s3://", "azfiles://", "http://", "https://", "/mnt/batch")
                    ):
                        if os.path.exists(model_path):
                            print(
                                f"[Model Staging] Staging local model: {os.path.basename(model_path)}"
                            )
                            remote_path = (
                                f"models/{os.path.basename(model_path)}"
                            )
                            submitter.azure_files_staging.upload_file(
                                local_path=model_path,
                                remote_path=remote_path,
                                show_progress=False,
                            )
                            azfiles_model_path = (
                                f"/mnt/batch/tasks/fsmounts/azfiles/{remote_path}"
                            )
                            task_default_params["model_path"] = (
                                azfiles_model_path
                            )
                            print(
                                f"[Model Staging] Updated model_path: {azfiles_model_path}"
                            )

                    # Stage slide model if it's a local path
                    slide_model_path = task_default_params.get("slide_model_path")
                    if slide_model_path and not slide_model_path.startswith(
                        ("s3://", "azfiles://", "http://", "https://", "/mnt/batch")
                    ):
                        if os.path.exists(slide_model_path):
                            print(
                                f"[Model Staging] Staging local slide model: {os.path.basename(slide_model_path)}"
                            )
                            remote_path = (
                                f"models/{os.path.basename(slide_model_path)}"
                            )
                            submitter.azure_files_staging.upload_file(
                                local_path=slide_model_path,
                                remote_path=remote_path,
                                show_progress=False,
                            )
                            azfiles_model_path = (
                                f"/mnt/batch/tasks/fsmounts/azfiles/{remote_path}"
                            )
                            task_default_params["slide_model_path"] = (
                                azfiles_model_path
                            )
                            print(
                                f"[Model Staging] Updated slide_model_path: {azfiles_model_path}"
                            )

                # Stage and submit tasks incrementally
                submitter.stage_and_submit_tasks_from_csv(
                    job_id=args.job_id,
                    csv_file=args.csv_manifest,
                    output_s3_prefix=args.output_prefix,
                    container_image=args.container_image,
                    models=models_list,
                    slide_models=slide_models_list,
                    remote_dir="slides",
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
        # Prepare model paths for single task
        task_model_paths = {}
        if model_paths.get("CTRANSPATH"):
            task_model_paths["prefilter_model_path"] = model_paths["CTRANSPATH"]
        if model_paths:
            # Try to find a model path
            for key, path in model_paths.items():
                if key != "CTRANSPATH" and not key.endswith("_SLIDE"):
                    task_model_paths["model_path"] = path
                    break
        # Look for slide model path by model type or fallback to "slide"
        slide_model_type = getattr(args, "slide_model_type", None) or config_defaults.get("slide_model_type")
        if slide_model_type and model_paths.get(slide_model_type):
            task_model_paths["slide_model_path"] = model_paths[slide_model_type]
        elif model_paths.get("slide"):
            # Fallback to generic "slide" key
            task_model_paths["slide_model_path"] = model_paths["slide"]

        submitter.submit_task(
            job_id=args.job_id,
            task_id=args.task_id,
            slide_path=args.slide_path,
            output_h5_path=args.output_h5_path,
            output_pt_path=args.output_pt_path,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            aws_region=args.aws_region,
            hf_token=args.hf_token,
            max_retry_count=args.max_retry_count,
            container_image=args.container_image,
            **task_model_paths,
        )
    else:
        print(
            "ERROR: Must specify either --csv-manifest or --task-id with --slide-path"
        )
        sys.exit(1)

    # Monitor if requested
    if args.monitor:
        submitter.monitor_tasks(job_id=args.job_id)

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
    if args.cleanup_staged_files:
        submitter.cleanup_staged_files()

    if args.delete_job:
        submitter.delete_job(job_id=args.job_id)

    if args.delete_pool:
        submitter.delete_pool(pool_id=args.pool_id)


if __name__ == "__main__":
    main()
