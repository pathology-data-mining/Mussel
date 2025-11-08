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

# Import model pre-download utility and Azure Files staging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
try:
    from model_predownload import pre_download_models, upload_models_to_s3
except ImportError:
    print("WARNING: Could not import model_predownload module. Pre-download features will be unavailable.")
    pre_download_models = None
    upload_models_to_s3 = None

try:
    from azure_files_staging import AzureFilesStaging
except ImportError:
    print("WARNING: Could not import azure_files_staging module. Azure Files staging features will be unavailable.")
    AzureFilesStaging = None

try:
    from config_loader import load_config, load_config_defaults, add_config_to_metadata
except ImportError:
    print("WARNING: Could not import config_loader module. YAML config support will be unavailable.")
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
    
    # Azure GPU VM family prefixes
    GPU_VM_PREFIXES = ['Standard_NC', 'Standard_ND', 'Standard_NV']

    def __init__(
        self,
        batch_account_name: str,
        batch_account_key: str,
        batch_account_url: str,
        storage_account_name: Optional[str] = None,
        storage_account_key: Optional[str] = None,
        azure_files_share_name: Optional[str] = None,
    ):
        """Initialize Azure Batch client."""
        credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
        self.batch_client = BatchServiceClient(credentials, batch_account_url)
        
        self.storage_account_name = storage_account_name
        self.storage_account_key = storage_account_key
        self.azure_files_share_name = azure_files_share_name
        
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
    
    def _should_set_intermediate_h5_path(self, aggregation_method: Optional[str]) -> bool:
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
        return aggregation_method is not None and aggregation_method != 'identity'

    def _should_use_batch_encoding(self, **kwargs) -> bool:
        """Determine if we should use batch slide encoding optimization.
        
        Batch encoding is beneficial when:
        1. Using model-based aggregation (aggregation_method="model")
        2. Using a slide encoder (slide_model_type is specified)
        3. Processing multiple slides
        
        Note: For Azure Batch, batch encoding optimization is implemented at the
        task script level (run_tessellate_extract_features.sh) by passing multiple
        slides to a single task when beneficial.
        """
        return (
            kwargs.get('aggregation_method') == 'model' and
            kwargs.get('slide_model_type') is not None
        )
        
        # Initialize Azure Files staging client if configured
        if storage_account_name and storage_account_key and azure_files_share_name and AzureFilesStaging:
            self.azure_files_staging = AzureFilesStaging(
                account_name=storage_account_name,
                account_key=storage_account_key,
                share_name=azure_files_share_name,
            )
        else:
            self.azure_files_staging = None

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
        publisher: str = "microsoft-dsvm",
        offer: str = "ubuntu-hpc",
        sku: str = "batch.node.ubuntu 22.04",
    ) -> None:
        """Create a pool of compute nodes with optional Azure Files mount and auto-scaling.
        
        Args:
            pool_id: Unique identifier for the pool
            vm_size: Azure VM size (e.g., Standard_NC24ads_A100_v4 for A100 GPU)
            node_count: Number of VMs in the pool (used as initial target for auto-scale or fixed count)
            container_image: Docker image to use
            use_gpu: Whether GPU support is intended (used for validation and logging)
            mount_azure_files: Whether to mount Azure Files share
            enable_auto_scale: Enable auto-scaling based on pending tasks
            min_node_count: Minimum number of nodes for auto-scaling (defaults to node_count)
            max_node_count: Maximum number of nodes for auto-scaling (required if enable_auto_scale=True)
            auto_scale_evaluation_interval: Auto-scale evaluation interval in minutes (default: 15)
            publisher: Azure VM image publisher (default: microsoft-dsvm)
            offer: Azure VM image offer (default: ubuntu-hpc)
            sku: Azure VM image SKU (default: batch.node.ubuntu 22.04)
        """
        print(f"Creating pool '{pool_id}'...")
        
        # Validate auto-scale configuration
        if enable_auto_scale:
            if max_node_count is None:
                raise ValueError("max_node_count is required when enable_auto_scale is True")
            if min_node_count is None:
                min_node_count = node_count
            if min_node_count > max_node_count:
                raise ValueError(f"min_node_count ({min_node_count}) cannot be greater than max_node_count ({max_node_count})")
            print(f"  Auto-scaling: Enabled (min: {min_node_count}, max: {max_node_count} nodes)")
        else:
            print(f"  Node count: {node_count} (fixed)")
        
        # Validate GPU configuration
        is_gpu_vm = any(vm_size.startswith(prefix) for prefix in self.GPU_VM_PREFIXES)
        
        if use_gpu and not is_gpu_vm:
            print(f"  WARNING: GPU support requested but VM size '{vm_size}' does not appear to be a GPU-enabled VM")
            print(f"  GPU-enabled VM sizes typically start with: {', '.join(self.GPU_VM_PREFIXES)}")
        elif not use_gpu and is_gpu_vm:
            print(f"  NOTE: GPU support disabled but VM size '{vm_size}' appears to be GPU-enabled")
        
        if use_gpu:
            print(f"  GPU support: Enabled (VM size: {vm_size})")
        else:
            print(f"  GPU support: Disabled (VM size: {vm_size})")

        # Container configuration
        container_conf = batchmodels.ContainerConfiguration(
            container_image_names=[container_image]
        )

        # VM configuration
        image_ref = batchmodels.ImageReference(
            publisher=publisher,
            offer=offer,
            sku=sku,
            version="latest",
        )

        vm_config = batchmodels.VirtualMachineConfiguration(
            image_reference=image_ref,
            container_configuration=container_conf,
            node_agent_sku_id=sku,
        )

        # Add Azure Files mount configuration if requested
        mount_config = None
        if mount_azure_files and self.azure_files_staging:
            print(f"  Configuring Azure Files mount: share '{self.azure_files_share_name}'")
            mount_config = [
                batchmodels.MountConfiguration(
                    azure_file_share_configuration=batchmodels.AzureFileShareConfiguration(
                        account_name=self.storage_account_name,
                        azure_file_url=f"https://{self.storage_account_name}.file.core.windows.net/{self.azure_files_share_name}",
                        account_key=self.storage_account_key,
                        relative_mount_path="azfiles",
                        mount_options="-o vers=3.0,dir_mode=0777,file_mode=0777,sec=ntlmssp",
                    )
                )
            ]

        # Pool configuration
        if enable_auto_scale:
            # Auto-scale formula based on pending tasks with unusable node handling
            # Based on: https://learn.microsoft.com/en-us/answers/questions/1699080/azure-batch-better-handle-unusable-nodes
            auto_scale_formula = f"""
                // Configuration
                startingNumberOfVMs = {min_node_count};
                maxNumberofVMs = {max_node_count};
                
                // Get pending tasks count
                pendingTaskSamplePercent = $PendingTasks.GetSamplePercent(180 * TimeInterval_Second);
                pendingTaskSamples = pendingTaskSamplePercent < 70 ? startingNumberOfVMs : avg($PendingTasks.GetSample(180 * TimeInterval_Second));
                
                // Calculate target based on pending tasks
                targetVMs = min(maxNumberofVMs, pendingTaskSamples);
                
                // Handle unusable nodes - add them to target to maintain capacity
                // Unusable nodes are those in unusable state that cannot run tasks
                unusableNodes = $CurrentDedicatedNodes - $RunningTasks - $ActiveTasks - $IdleNodes;
                
                // Ensure we have enough nodes to handle workload even with unusable nodes
                $TargetDedicatedNodes = min(maxNumberofVMs, max(startingNumberOfVMs, targetVMs + max(0, unusableNodes)));
            """
            
            pool = batchmodels.PoolAddParameter(
                id=pool_id,
                virtual_machine_configuration=vm_config,
                vm_size=vm_size,
                enable_auto_scale=True,
                auto_scale_formula=auto_scale_formula,
                auto_scale_evaluation_interval=datetime.timedelta(minutes=auto_scale_evaluation_interval),
                mount_configuration=mount_config,
            )
        else:
            pool = batchmodels.PoolAddParameter(
                id=pool_id,
                virtual_machine_configuration=vm_config,
                vm_size=vm_size,
                target_dedicated_nodes=node_count,
                enable_auto_scale=False,
                mount_configuration=mount_config,
            )

        try:
            self.batch_client.pool.add(pool)
            print(f"Pool '{pool_id}' created successfully")
            if mount_config:
                print(f"  Azure Files share '{self.azure_files_share_name}' will be mounted at /mnt/batch/tasks/fsmounts/azfiles")
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
        slide_path: str,
        output_h5_path: str,
        output_pt_path: str,
        intermediate_h5_path: Optional[str] = None,
        aggregation_method: str = "identity",
        slide_model_type: Optional[str] = None,
        classifier_pkl: Optional[str] = None,
        classifier_threshold: float = 0.75,
        prefilter_model_type: str = "CTRANSPATH",
        prefilter_model_path: Optional[str] = None,
        postfilter_model_type: Optional[str] = None,
        postfilter_model_path: Optional[str] = None,
        postfilter_model_types: Optional[str] = None,
        slide_model_path: Optional[str] = None,
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
    ) -> None:
        """Submit a tessellate-extract-features task to Azure Batch."""
        print(f"Submitting task '{task_id}' to job '{job_id}'...")

        # Build environment variables
        env_vars = [
            batchmodels.EnvironmentSetting("SLIDE_PATH", slide_path),
            batchmodels.EnvironmentSetting("OUTPUT_H5_PATH", output_h5_path),
            batchmodels.EnvironmentSetting("OUTPUT_PT_PATH", output_pt_path),
            batchmodels.EnvironmentSetting("PREFILTER_MODEL_TYPE", prefilter_model_type),
            batchmodels.EnvironmentSetting("NUM_WORKERS", str(num_workers)),
            batchmodels.EnvironmentSetting("BATCH_SIZE", str(batch_size)),
            batchmodels.EnvironmentSetting("USE_GPU", str(use_gpu).lower()),
            batchmodels.EnvironmentSetting("KEEP_INTERMEDIATE_FILES", str(keep_intermediate_files).lower()),
            batchmodels.EnvironmentSetting("AGGREGATION_METHOD", aggregation_method),
        ]
        
        # SegConfig group or individual parameters
        if seg_config_group:
            env_vars.append(batchmodels.EnvironmentSetting("SEG_CONFIG_GROUP", seg_config_group))
        
        # Individual SegConfig parameters (only set if provided)
        if segment_threshold is not None:
            env_vars.append(batchmodels.EnvironmentSetting("SEGMENT_THRESHOLD", str(segment_threshold)))
        if patch_size is not None:
            env_vars.append(batchmodels.EnvironmentSetting("PATCH_SIZE", str(patch_size)))
        if step_size is not None:
            env_vars.append(batchmodels.EnvironmentSetting("STEP_SIZE", str(step_size)))
        if mpp is not None:
            env_vars.append(batchmodels.EnvironmentSetting("MPP", str(mpp)))
        if seg_level is not None:
            env_vars.append(batchmodels.EnvironmentSetting("SEG_LEVEL", str(seg_level)))
        if segment_max_value is not None:
            env_vars.append(batchmodels.EnvironmentSetting("SEGMENT_MAX_VALUE", str(segment_max_value)))
        if median_blur_ksize is not None:
            env_vars.append(batchmodels.EnvironmentSetting("MEDIAN_BLUR_KSIZE", str(median_blur_ksize)))
        if morphology_ex_kernel is not None:
            env_vars.append(batchmodels.EnvironmentSetting("MORPHOLOGY_EX_KERNEL", str(morphology_ex_kernel)))
        if ref_patch_size is not None:
            env_vars.append(batchmodels.EnvironmentSetting("REF_PATCH_SIZE", str(ref_patch_size)))
        if use_otsu is not None:
            env_vars.append(batchmodels.EnvironmentSetting("USE_OTSU", str(use_otsu).lower()))
        if tissue_area_threshold is not None:
            env_vars.append(batchmodels.EnvironmentSetting("TISSUE_AREA_THRESHOLD", str(tissue_area_threshold)))
        if hole_area_threshold is not None:
            env_vars.append(batchmodels.EnvironmentSetting("HOLE_AREA_THRESHOLD", str(hole_area_threshold)))
        if max_num_holes is not None:
            env_vars.append(batchmodels.EnvironmentSetting("MAX_NUM_HOLES", str(max_num_holes)))

        if intermediate_h5_path:
            env_vars.append(batchmodels.EnvironmentSetting("INTERMEDIATE_H5_PATH", intermediate_h5_path))

        if slide_model_type:
            env_vars.append(batchmodels.EnvironmentSetting("SLIDE_MODEL_TYPE", slide_model_type))

        if classifier_pkl:
            env_vars.append(batchmodels.EnvironmentSetting("CLASSIFIER_PKL", classifier_pkl))
            env_vars.append(batchmodels.EnvironmentSetting("CLASSIFIER_THRESHOLD", str(classifier_threshold)))

        # Handle postfilter models - either single or multiple
        if postfilter_model_types:
            # Multiple models - comma-separated list
            env_vars.append(batchmodels.EnvironmentSetting("POSTFILTER_MODEL_TYPES", postfilter_model_types))
        elif postfilter_model_type:
            # Single model
            env_vars.append(batchmodels.EnvironmentSetting("POSTFILTER_MODEL_TYPE", postfilter_model_type))
        
        # Add model paths if provided
        if prefilter_model_path:
            env_vars.append(batchmodels.EnvironmentSetting("PREFILTER_MODEL_PATH", prefilter_model_path))
        if postfilter_model_path:
            env_vars.append(batchmodels.EnvironmentSetting("POSTFILTER_MODEL_PATH", postfilter_model_path))
        if slide_model_path:
            env_vars.append(batchmodels.EnvironmentSetting("SLIDE_MODEL_PATH", slide_model_path))

        if hf_token:
            env_vars.append(batchmodels.EnvironmentSetting("HF_TOKEN", hf_token))

        # Add AWS credentials if provided (for S3 access)
        if aws_access_key_id:
            env_vars.append(batchmodels.EnvironmentSetting("AWS_ACCESS_KEY_ID", aws_access_key_id))
        if aws_secret_access_key:
            env_vars.append(batchmodels.EnvironmentSetting("AWS_SECRET_ACCESS_KEY", aws_secret_access_key))
        if aws_region:
            env_vars.append(batchmodels.EnvironmentSetting("AWS_DEFAULT_REGION", aws_region))
        if aws_endpoint_url:
            env_vars.append(batchmodels.EnvironmentSetting("AWS_ENDPOINT_URL", aws_endpoint_url))
        
        # Add Azure Files cleanup settings if enabled
        if cleanup_staged_file:
            env_vars.append(batchmodels.EnvironmentSetting("CLEANUP_STAGED_FILE", "true"))
            if self.storage_account_name:
                env_vars.append(batchmodels.EnvironmentSetting("AZURE_STORAGE_ACCOUNT", self.storage_account_name))
            if self.storage_account_key:
                env_vars.append(batchmodels.EnvironmentSetting("AZURE_STORAGE_KEY", self.storage_account_key))
            if self.azure_files_share_name:
                env_vars.append(batchmodels.EnvironmentSetting("AZURE_FILES_SHARE", self.azure_files_share_name))

        # Container settings
        container_settings = batchmodels.TaskContainerSettings(
            image_name=container_image,
            container_run_options="--rm --gpus all" if use_gpu else "--rm",
        )

        # Task command - run the task script
        task_command = "/bin/bash /app/scripts/azure_batch/run_tessellate_extract_features.sh"

        # Task constraints with retry configuration
        task_constraints = batchmodels.TaskConstraints(
            max_task_retry_count=max_retry_count
        )

        # Create task
        task = batchmodels.TaskAddParameter(
            id=task_id,
            command_line=task_command,
            container_settings=container_settings,
            environment_settings=env_vars,
            constraints=task_constraints,
        )

        try:
            self.batch_client.task.add(job_id, task)
            
            # Store task metadata for failure tracking
            self.task_metadata[task_id] = {
                'slide_path': slide_path,
                'output_h5_path': output_h5_path,
                'output_pt_path': output_pt_path,
            }
            if intermediate_h5_path:
                self.task_metadata[task_id]['intermediate_h5_path'] = intermediate_h5_path
            
            print(f"Task '{task_id}' submitted successfully (max retries: {max_retry_count})")
        except batchmodels.BatchErrorException as e:
            print(f"Error submitting task: {e}")
            raise

    def stage_and_submit_tasks_from_csv(
        self,
        job_id: str,
        csv_file: str,
        output_dir: str = "/mnt/output",
        output_s3_prefix: Optional[str] = None,
        container_image: str = "mskmind/mussel:latest-torch-gpu",
        postfilter_models: Optional[List[str]] = None,
        remote_dir: str = "slides",
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
            output_dir: Local output directory for results (default: /mnt/output)
            output_s3_prefix: S3 prefix for outputs (e.g., s3://bucket/results/)
            container_image: Docker image to use
            postfilter_models: List of postfilter model types to run sequentially in same task
            remote_dir: Remote directory for slides in Azure Files
            **default_params: Default parameters for all tasks (e.g., prefilter_model_type, batch_size)
        """
        if not self.azure_files_staging:
            raise ValueError("Azure Files staging not configured. Provide storage account details and share name.")
        
        print(f"Loading task manifest from '{csv_file}'...")

        # Get prefilter model type (used for directory organization when single model)
        prefilter_model = default_params.get('prefilter_model_type', 'CTRANSPATH')
        
        # Determine model type for directory structure
        if postfilter_models and len(postfilter_models) > 1:
            # Multiple models - use first model for base directory, but actual paths will be model-specific
            model_type = postfilter_models[0]
            print(f"Will process each slide with {len(postfilter_models)} postfilter models sequentially: {', '.join(postfilter_models)}")
        elif postfilter_models and len(postfilter_models) == 1:
            model_type = postfilter_models[0]
        else:
            # Single model from default_params or prefilter
            model_type = default_params.get('postfilter_model_type', prefilter_model)
        
        # Read CSV and process slides one by one
        tasks_submitted = 0
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            slides = list(reader)
        
        print(f"[Azure Files] Staging and submitting {len(slides)} slides...")
        print(f"[Azure Files] Tasks will start processing as slides are staged")
        
        for idx, row in enumerate(slides, 1):
            slide_id = row['slide_id']
            slide_path = row['slide_path']
            
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
            if output_s3_prefix:
                base_prefix = output_s3_prefix.rstrip('/')
                output_h5_path = f"{base_prefix}/{model_type}/h5/{slide_id}_features.h5"
                output_pt_path = f"{base_prefix}/{model_type}/pt/{slide_id}_features.pt"
                # Only set intermediate_h5_path if aggregation method requires it
                if self._should_set_intermediate_h5_path(default_params.get('aggregation_method')):
                    intermediate_h5_path = f"{base_prefix}/{model_type}/tile_h5/{slide_id}_tile_features.h5"
                else:
                    intermediate_h5_path = None
            else:
                output_h5_path = f"{output_dir}/{model_type}/h5/{slide_id}_features.h5"
                output_pt_path = f"{output_dir}/{model_type}/pt/{slide_id}_features.pt"
                # Only set intermediate_h5_path if aggregation method requires it
                if self._should_set_intermediate_h5_path(default_params.get('aggregation_method')):
                    intermediate_h5_path = f"{output_dir}/{model_type}/tile_h5/{slide_id}_tile_features.h5"
                else:
                    intermediate_h5_path = None
            
            # Merge with default parameters
            merged_config = {**default_params}
            merged_config['task_id'] = task_id
            merged_config['slide_path'] = azfiles_path
            merged_config['output_h5_path'] = output_h5_path
            merged_config['output_pt_path'] = output_pt_path
            if intermediate_h5_path:
                merged_config['intermediate_h5_path'] = intermediate_h5_path
            
            # Add postfilter models as comma-separated list if multiple
            if postfilter_models and len(postfilter_models) > 1:
                merged_config['postfilter_model_types'] = ','.join(postfilter_models)
            elif postfilter_models and len(postfilter_models) == 1:
                merged_config['postfilter_model_type'] = postfilter_models[0]
            
            # Submit task immediately
            print(f"[{idx}/{len(slides)}] Submitting task for {slide_id}...")
            self.submit_task(
                job_id=job_id,
                task_id=merged_config['task_id'],
                slide_path=merged_config['slide_path'],
                output_h5_path=merged_config['output_h5_path'],
                output_pt_path=merged_config['output_pt_path'],
                intermediate_h5_path=merged_config.get('intermediate_h5_path'),
                aggregation_method=merged_config.get("aggregation_method", "identity"),
                slide_model_type=merged_config.get("slide_model_type"),
                classifier_pkl=merged_config.get("classifier_pkl"),
                classifier_threshold=merged_config.get("classifier_threshold", 0.75),
                prefilter_model_type=merged_config.get("prefilter_model_type", "CTRANSPATH"),
                postfilter_model_type=merged_config.get("postfilter_model_type"),
                postfilter_model_types=merged_config.get("postfilter_model_types"),
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
                keep_intermediate_files=merged_config.get("keep_intermediate_files", False),
                hf_token=merged_config.get("hf_token"),
                aws_access_key_id=merged_config.get("aws_access_key_id"),
                aws_secret_access_key=merged_config.get("aws_secret_access_key"),
                aws_region=merged_config.get("aws_region"),
                max_retry_count=merged_config.get("max_retry_count", 3),
                container_image=container_image,
                cleanup_staged_file=True,  # Enable per-task cleanup for staged files
            )
            tasks_submitted += 1
        
        print(f"\n[Azure Files] Staged and submitted {tasks_submitted} tasks")

    def submit_tasks_from_csv(
        self,
        job_id: str,
        csv_file: str,
        output_dir: str = "/mnt/output",
        output_s3_prefix: Optional[str] = None,
        container_image: str = "mskmind/mussel:latest-torch-gpu",
        postfilter_models: Optional[List[str]] = None,
        staged_slide_paths: Optional[Dict[str, str]] = None,
        distributed_slide_batch_size: int = 1,
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
            output_dir: Local output directory for results (default: /mnt/output)
            output_s3_prefix: S3 prefix for outputs (e.g., s3://bucket/results/)
            container_image: Docker image to use
            postfilter_models: List of postfilter model types to run sequentially in same task
            staged_slide_paths: Optional dict mapping slide_id to staged Azure Files paths
            distributed_slide_batch_size: Number of slides to group per task for batch encoding (default: 1).
                When > 1 and using slide-level model aggregation, slides are grouped into batches
                to optimize slide encoder loading. Recommended: 8-16 for GIGAPATH_SLIDE/TITAN_SLIDE.
            **default_params: Default parameters for all tasks (e.g., prefilter_model_type, batch_size)
        """
        print(f"Loading task manifest from '{csv_file}'...")

        # Get prefilter model type (used for directory organization when single model)
        prefilter_model = default_params.get('prefilter_model_type', 'CTRANSPATH')
        
        # Determine model type for directory structure
        if postfilter_models and len(postfilter_models) > 1:
            # Multiple models - use first model for base directory, but actual paths will be model-specific
            model_type = postfilter_models[0]
            print(f"Will process each slide with {len(postfilter_models)} postfilter models sequentially: {', '.join(postfilter_models)}")
        elif postfilter_models and len(postfilter_models) == 1:
            model_type = postfilter_models[0]
        else:
            # Single model from default_params or prefilter
            model_type = default_params.get('postfilter_model_type', prefilter_model)
        
        tasks = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                slide_id = row['slide_id']
                slide_path = row['slide_path']
                
                # Use staged path if available
                if staged_slide_paths and slide_id in staged_slide_paths:
                    slide_path = staged_slide_paths[slide_id]
                    print(f"  Using staged path for {slide_id}: {slide_path}")
                
                # Create ONE task per slide (models run sequentially within the task)
                task_id = slide_id
                
                # For multi-model, base output paths use the first model's directory
                # The bash script will handle creating model-specific subdirectories
                if output_s3_prefix:
                    base_prefix = output_s3_prefix.rstrip('/')
                    output_h5_path = f"{base_prefix}/{model_type}/h5/{slide_id}_features.h5"
                    output_pt_path = f"{base_prefix}/{model_type}/pt/{slide_id}_features.pt"
                    # Only set intermediate_h5_path if aggregation method requires it
                    if self._should_set_intermediate_h5_path(default_params.get('aggregation_method')):
                        intermediate_h5_path = f"{base_prefix}/{model_type}/tile_h5/{slide_id}_tile_features.h5"
                    else:
                        intermediate_h5_path = None
                else:
                    output_h5_path = f"{output_dir}/{model_type}/h5/{slide_id}_features.h5"
                    output_pt_path = f"{output_dir}/{model_type}/pt/{slide_id}_features.pt"
                    # Only set intermediate_h5_path if aggregation method requires it
                    if self._should_set_intermediate_h5_path(default_params.get('aggregation_method')):
                        intermediate_h5_path = f"{output_dir}/{model_type}/tile_h5/{slide_id}_tile_features.h5"
                    else:
                        intermediate_h5_path = None
                
                # Create task config
                task_config = {
                    'task_id': task_id,
                    'slide_path': slide_path,
                    'output_h5_path': output_h5_path,
                    'output_pt_path': output_pt_path,
                }
                
                # Only add intermediate_h5_path if it's set
                if intermediate_h5_path:
                    task_config['intermediate_h5_path'] = intermediate_h5_path
                
                # Add postfilter models as comma-separated list if multiple
                if postfilter_models and len(postfilter_models) > 1:
                    task_config['postfilter_model_types'] = ','.join(postfilter_models)
                elif postfilter_models and len(postfilter_models) == 1:
                    task_config['postfilter_model_type'] = postfilter_models[0]
                
                tasks.append(task_config)

        print(f"Submitting {len(tasks)} tasks from CSV manifest...")

        for task_config in tasks:
            # Merge with default parameters
            merged_config = {**default_params, **task_config}
            
            # Normalize empty string to None for intermediate_h5_path
            intermediate_h5_path = merged_config.get('intermediate_h5_path') or None

            self.submit_task(
                job_id=job_id,
                task_id=merged_config['task_id'],
                slide_path=merged_config['slide_path'],
                output_h5_path=merged_config['output_h5_path'],
                output_pt_path=merged_config['output_pt_path'],
                intermediate_h5_path=intermediate_h5_path,
                aggregation_method=merged_config.get("aggregation_method", "identity"),
                slide_model_type=merged_config.get("slide_model_type"),
                classifier_pkl=merged_config.get("classifier_pkl"),
                classifier_threshold=merged_config.get("classifier_threshold", 0.75),
                prefilter_model_type=merged_config.get("prefilter_model_type", "CTRANSPATH"),
                postfilter_model_type=merged_config.get("postfilter_model_type"),
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
                keep_intermediate_files=merged_config.get("keep_intermediate_files", False),
                hf_token=merged_config.get("hf_token"),
                aws_access_key_id=merged_config.get("aws_access_key_id"),
                aws_secret_access_key=merged_config.get("aws_secret_access_key"),
                aws_region=merged_config.get("aws_region"),
                max_retry_count=merged_config.get("max_retry_count", 3),
                container_image=container_image,
            )
            
            # Store task configuration in metadata (excluding secrets)
            if add_config_to_metadata:
                add_config_to_metadata(self.task_metadata, merged_config, merged_config['task_id'])

    def monitor_tasks(self, job_id: str, poll_interval: int = 30) -> None:
        """Monitor task progress."""
        print(f"Monitoring tasks in job '{job_id}'...")
        print("Press Ctrl+C to stop monitoring (tasks will continue running)")

        try:
            while True:
                tasks = list(self.batch_client.task.list(job_id))
                
                total = len(tasks)
                completed = sum(1 for t in tasks if t.state == batchmodels.TaskState.completed)
                active = sum(1 for t in tasks if t.state == batchmodels.TaskState.active)
                running = sum(1 for t in tasks if t.state == batchmodels.TaskState.running)
                
                print(f"\n[{datetime.datetime.now()}] Status: {completed}/{total} completed, "
                      f"{running} running, {active} active")

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
                'task_id': task_id,
                'state': str(task.state),
                'exit_code': task.execution_info.exit_code if task.execution_info else 'N/A',
            }
            
            # If we have metadata, include the original task configuration
            if task_metadata and task_id in task_metadata:
                task_info.update(task_metadata[task_id])
            else:
                # Extract from environment variables if available
                if task.environment_settings:
                    for env_var in task.environment_settings:
                        if env_var.name == 'SLIDE_PATH':
                            task_info['slide_path'] = env_var.value
            
            failed_data.append(task_info)
        
        # Write to CSV
        if failed_data:
            # Determine CSV headers from first failed task
            headers = list(failed_data[0].keys())
            
            with open(output_file, 'w', newline='') as f:
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
                'task_id': task_id,
                'state': 'completed',
                'exit_code': 0,
            }
            
            # If we have metadata, include the output paths
            if task_metadata and task_id in task_metadata:
                metadata = task_metadata[task_id]
                task_info['slide_path'] = metadata.get('slide_path', '')
                task_info['output_h5_path'] = metadata.get('output_h5_path', '')
                task_info['output_pt_path'] = metadata.get('output_pt_path', '')
                
                # Add intermediate path if present
                if 'intermediate_h5_path' in metadata:
                    task_info['intermediate_h5_path'] = metadata.get('intermediate_h5_path', '')
                
                # Add configuration parameters if present (excluding secrets)
                if 'config' in metadata:
                    config_data = metadata['config']
                    # Flatten configuration into task_info with 'config_' prefix
                    # Note: Nested dictionaries and lists are skipped for CSV compatibility
                    for key, value in config_data.items():
                        # Skip nested dictionaries and lists for CSV simplicity
                        if not isinstance(value, (dict, list)):
                            task_info[f'config_{key}'] = value
                
                # Extract model type from output path
                output_h5 = metadata.get('output_h5_path', '')
                if '/' in output_h5:
                    parts = output_h5.split('/')
                    # Look for model type in path (e.g., /CTRANSPATH/h5/)
                    for i, part in enumerate(parts):
                        if i < len(parts) - 1 and parts[i + 1] in ['h5', 'pt', 'tile_h5']:
                            task_info['model_type'] = part
                            break
                
                # Extract file type from path
                if output_h5.endswith('.h5'):
                    if 'tile' in output_h5:
                        task_info['file_type'] = 'tile_h5'
                    else:
                        task_info['file_type'] = 'h5'
            else:
                # Extract from environment variables if available
                if task.environment_settings:
                    for env_var in task.environment_settings:
                        if env_var.name == 'SLIDE_PATH':
                            task_info['slide_path'] = env_var.value
                        elif env_var.name == 'OUTPUT_H5_PATH':
                            task_info['output_h5_path'] = env_var.value
                        elif env_var.name == 'OUTPUT_PT_PATH':
                            task_info['output_pt_path'] = env_var.value
                        elif env_var.name == 'INTERMEDIATE_H5_PATH':
                            task_info['intermediate_h5_path'] = env_var.value
                        elif env_var.name == 'PREFILTER_MODEL_TYPE':
                            task_info['model_type'] = env_var.value
            
            manifest_data.append(task_info)
        
        # Write to CSV
        if manifest_data:
            # Determine CSV headers from first task
            headers = list(manifest_data[0].keys())
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(manifest_data)
            
            print(f"Saved results manifest with {len(manifest_data)} successful tasks to '{output_file}'")
        
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
            raise ValueError("Azure Files staging not configured. Provide storage account details and share name.")
        
        staged_paths = {}
        print(f"[Azure Files] Staging {len(slides)} slides to Azure Files share '{self.azure_files_share_name}'...")
        
        for slide in slides:
            slide_id = slide['slide_id']
            slide_path = slide['slide_path']
            
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
    
    # Azure credentials (can be provided via command-line, environment variables, or config file)
    parser.add_argument("--batch-account-name", help="Azure Batch account name (or set AZURE_BATCH_ACCOUNT_NAME env var)")
    parser.add_argument("--batch-account-key", help="Azure Batch account key (or set AZURE_BATCH_ACCOUNT_KEY env var)")
    parser.add_argument("--batch-account-url", help="Azure Batch account URL (or set AZURE_BATCH_ACCOUNT_URL env var)")
    parser.add_argument("--storage-account-name", help="Azure Storage account name (optional, or set AZURE_STORAGE_ACCOUNT_NAME env var)")
    parser.add_argument("--storage-account-key", help="Azure Storage account key (optional, or set AZURE_STORAGE_ACCOUNT_KEY env var)")
    
    # Azure Files staging configuration
    parser.add_argument("--azure-files-share-name", help="Azure Files share name for staging files")
    parser.add_argument("--stage-to-azure-files", action="store_true", 
                        help="Stage input files to Azure Files before processing")
    parser.add_argument("--mount-azure-files", action="store_true",
                        help="Mount Azure Files share to batch pool nodes")
    parser.add_argument("--cleanup-staged-files", action="store_true",
                        help="Clean up staged files from Azure Files after processing")
    
    # Pool configuration
    parser.add_argument("--pool-id", help="Pool ID (can be specified in config file)")
    parser.add_argument("--create-pool", action="store_true", help="Create pool if it doesn't exist")
    
    # Default values for pool parameters (used for detecting if config should override)
    DEFAULT_VM_SIZE = "Standard_NC24ads_A100_v4"
    DEFAULT_NODE_COUNT = 1
    DEFAULT_CONTAINER_IMAGE = "mskmind/mussel:latest-torch-gpu"
    DEFAULT_AUTO_SCALE_INTERVAL = 15
    
    parser.add_argument("--vm-size", default=DEFAULT_VM_SIZE, help="VM size for pool nodes")
    parser.add_argument("--node-count", type=int, default=DEFAULT_NODE_COUNT, 
                        help="Number of nodes in pool (or initial/min count for auto-scaling)")
    parser.add_argument("--use-gpu", action="store_true", default=True, 
                        help="Enable GPU support for pool nodes (default: True)")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                        help="Disable GPU support for pool nodes")
    parser.add_argument("--container-image", default=DEFAULT_CONTAINER_IMAGE, 
                        help="Docker container image")
    parser.add_argument("--enable-auto-scale", action="store_true",
                        help="Enable auto-scaling based on pending tasks")
    parser.add_argument("--min-node-count", type=int,
                        help="Minimum number of nodes for auto-scaling (defaults to --node-count)")
    parser.add_argument("--max-node-count", type=int,
                        help="Maximum number of nodes for auto-scaling (required if --enable-auto-scale)")
    parser.add_argument("--auto-scale-evaluation-interval", type=int, default=DEFAULT_AUTO_SCALE_INTERVAL,
                        help="Auto-scale evaluation interval in minutes (default: 15)")
    
    # VM image configuration
    parser.add_argument("--publisher", default="microsoft-dsvm",
                        help="Azure VM image publisher (default: microsoft-dsvm)")
    parser.add_argument("--offer", default="ubuntu-hpc",
                        help="Azure VM image offer (default: ubuntu-hpc)")
    parser.add_argument("--sku", default="batch.node.ubuntu 22.04",
                        help="Azure VM image SKU and node agent SKU ID (default: batch.node.ubuntu 22.04)")
    
    # Job configuration
    parser.add_argument("--job-id", help="Job ID (can be specified in config file)")
    parser.add_argument("--create-job", action="store_true", help="Create job")
    
    # Task configuration
    parser.add_argument("--config-file", "--config", dest="config_file", 
                        help="Configuration file with parameters (JSON or YAML format). "
                        "Can be used alone with task definitions, or with --csv-manifest to provide default parameters.")
    parser.add_argument("--csv-manifest", help="CSV manifest file with slide_id,slide_path columns. "
                        "Can be used with --config-file or --config to load parameters from config.")
    parser.add_argument("--output-dir", default="/mnt/output", help="Output directory for results (when using CSV)")
    parser.add_argument("--output-s3-prefix", help="S3 prefix for outputs (e.g., s3://bucket/results/)")
    parser.add_argument("--postfilter-models", help="Comma-separated list of postfilter model types to run (e.g., CTRANSPATH,CLIP,VIRCHOW)")
    parser.add_argument("--distributed-slide-batch-size", type=int, default=1,
                        help="Number of slides to group per distributed task for batch encoding optimization (default: 1). "
                             "When > 1 and using slide-level model aggregation (e.g., GIGAPATH_SLIDE), slides are grouped "
                             "into batches to optimize slide encoder loading. Recommended: 8-16 for better efficiency. "
                             "Note: Not applicable when using --stage-to-azure-files (incremental staging).")
    parser.add_argument("--task-id", help="Single task ID")
    parser.add_argument("--slide-path", help="Path to slide file (for single task, can be s3://)")
    parser.add_argument("--output-h5-path", help="Output H5 path (for single task, can be s3://)")
    parser.add_argument("--output-pt-path", help="Output PT path (for single task, can be s3://)")
    
    # AWS credentials for S3 access (can be provided via command-line, environment variables, or config file)
    parser.add_argument("--aws-access-key-id", help="AWS access key ID for S3 (or set AWS_ACCESS_KEY_ID env var)")
    parser.add_argument("--aws-secret-access-key", help="AWS secret access key for S3 (or set AWS_SECRET_ACCESS_KEY env var)")
    parser.add_argument("--aws-region", default="us-east-1", help="AWS region (or set AWS_DEFAULT_REGION env var)")
    parser.add_argument("--aws-endpoint-url", help="Custom S3 endpoint URL (e.g., for MinIO or Ceph, or set AWS_ENDPOINT_URL env var)")
    
    # HuggingFace token for model downloads (can be provided via command-line, environment variables, or config file)
    parser.add_argument("--hf-token", help="HuggingFace token for model downloads (or set HF_TOKEN env var)")
    
    # Model pre-download configuration
    parser.add_argument("--pre-download-models", action="store_true", default=True,
                        help="Pre-download models before job submission (default: True for batch jobs)")
    parser.add_argument("--no-pre-download-models", dest="pre_download_models", action="store_false",
                        help="Disable model pre-download")
    parser.add_argument("--model-cache-dir", default="./model_cache",
                        help="Local directory to cache models (default: ./model_cache)")
    parser.add_argument("--model-s3-prefix", help="S3 prefix for uploaded models (default: use output-s3-prefix/models/)")
    parser.add_argument("--prefilter-model-path", help="Path to prefilter model weights (local or s3://)")
    parser.add_argument("--postfilter-model-path", help="Path to postfilter model weights (local or s3://)")
    parser.add_argument("--slide-model-path", help="Path to slide encoder model weights (local or s3://)")
    
    # Retry configuration
    parser.add_argument("--max-retry-count", type=int, default=3, help="Maximum number of retry attempts for failed tasks (default: 3)")
    parser.add_argument("--save-failed-tasks", help="Save failed tasks to CSV file for resubmission")
    parser.add_argument("--generate-manifest", help="Generate manifest of successful result files to CSV")
    
    # Monitoring and cleanup
    parser.add_argument("--monitor", action="store_true", help="Monitor task progress")
    parser.add_argument("--delete-job", action="store_true", help="Delete job after completion")
    parser.add_argument("--delete-pool", action="store_true", 
                        help="Delete pool after completion. When used with --monitor, "
                        "the pool will be deleted after all tasks complete. "
                        "Otherwise, it will be deleted immediately.")
    
    args = parser.parse_args()

    # Load config file early if provided, to check for model paths before pre-download
    config_defaults = {}
    if args.config_file and load_config_defaults:
        try:
            config_defaults = load_config_defaults(args.config_file, backend='azure')
        except Exception as e:
            print(f"WARNING: Failed to load config file: {e}")
            config_defaults = {}
    
    # Priority order for credentials: CLI args > Environment variables > Config file
    # This allows flexible credential management for CI/CD and local development
    
    # Azure Batch credentials
    if not args.batch_account_name:
        args.batch_account_name = os.environ.get('AZURE_BATCH_ACCOUNT_NAME') or config_defaults.get('batch_account_name')
    
    if not args.batch_account_key:
        args.batch_account_key = os.environ.get('AZURE_BATCH_ACCOUNT_KEY') or config_defaults.get('batch_account_key')
    
    if not args.batch_account_url:
        args.batch_account_url = os.environ.get('AZURE_BATCH_ACCOUNT_URL') or config_defaults.get('batch_account_url')
    
    # Azure Storage credentials
    if not args.storage_account_name:
        args.storage_account_name = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME') or config_defaults.get('storage_account_name')
    
    if not args.storage_account_key:
        args.storage_account_key = os.environ.get('AZURE_STORAGE_ACCOUNT_KEY') or config_defaults.get('storage_account_key')
    
    # AWS credentials for S3 access
    if not args.aws_access_key_id:
        args.aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID') or config_defaults.get('aws_access_key_id')
    
    if not args.aws_secret_access_key:
        args.aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY') or config_defaults.get('aws_secret_access_key')
    
    if args.aws_region == "us-east-1":  # Check if it's the default value
        env_region = os.environ.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_REGION')
        if env_region:
            args.aws_region = env_region
        elif 'aws_region' in config_defaults:
            args.aws_region = config_defaults['aws_region']
    
    if not args.aws_endpoint_url:
        args.aws_endpoint_url = os.environ.get('AWS_ENDPOINT_URL') or config_defaults.get('aws_endpoint_url')
    
    # HuggingFace token
    if not args.hf_token:
        args.hf_token = os.environ.get('HF_TOKEN') or config_defaults.get('hf_token')
    
    # Validate required credentials
    if not args.batch_account_name:
        print("ERROR: --batch-account-name is required (or set AZURE_BATCH_ACCOUNT_NAME env var or specify in config file)")
        sys.exit(1)
    
    if not args.batch_account_key:
        print("ERROR: --batch-account-key is required (or set AZURE_BATCH_ACCOUNT_KEY env var or specify in config file)")
        sys.exit(1)
    
    if not args.batch_account_url:
        print("ERROR: --batch-account-url is required (or set AZURE_BATCH_ACCOUNT_URL env var or specify in config file)")
        sys.exit(1)
    
    # Apply Azure-specific parameters from config file if provided
    # Command-line arguments take precedence over config file values
    if config_defaults:
        # Pool and job IDs
        if not args.pool_id and 'pool_id' in config_defaults:
            args.pool_id = config_defaults['pool_id']
        
        if not args.job_id and 'job_id' in config_defaults:
            args.job_id = config_defaults['job_id']
        
        # Storage account parameters
        if not args.storage_account_name and 'storage_account_name' in config_defaults:
            args.storage_account_name = config_defaults['storage_account_name']
        
        # Azure Files share name
        if not args.azure_files_share_name and 'azure_files_share_name' in config_defaults:
            args.azure_files_share_name = config_defaults['azure_files_share_name']
        
        # Boolean flags (only set if not already set via command-line)
        if not args.create_pool and config_defaults.get('create_pool'):
            args.create_pool = config_defaults['create_pool']
        
        if not args.create_job and config_defaults.get('create_job'):
            args.create_job = config_defaults['create_job']
        
        if not args.mount_azure_files and config_defaults.get('mount_azure_files'):
            args.mount_azure_files = config_defaults['mount_azure_files']
        
        # Container image - only override if the default value is still being used
        if args.container_image == DEFAULT_CONTAINER_IMAGE and 'container_image' in config_defaults:
            args.container_image = config_defaults['container_image']
        
        # VM size - only override if default is being used
        if args.vm_size == DEFAULT_VM_SIZE and 'vm_size' in config_defaults:
            args.vm_size = config_defaults['vm_size']
        
        # Node count - only override if default is being used
        if args.node_count == DEFAULT_NODE_COUNT and 'node_count' in config_defaults:
            args.node_count = config_defaults['node_count']
        
        # Auto-scaling parameters
        if not args.enable_auto_scale and config_defaults.get('enable_auto_scale'):
            args.enable_auto_scale = config_defaults['enable_auto_scale']
        
        if args.min_node_count is None and 'min_node_count' in config_defaults:
            args.min_node_count = config_defaults['min_node_count']
        
        if args.max_node_count is None and 'max_node_count' in config_defaults:
            args.max_node_count = config_defaults['max_node_count']
        
        if args.auto_scale_evaluation_interval == DEFAULT_AUTO_SCALE_INTERVAL and 'auto_scale_evaluation_interval' in config_defaults:
            args.auto_scale_evaluation_interval = config_defaults['auto_scale_evaluation_interval']
        
        # VM image configuration parameters
        if args.publisher == "microsoft-dsvm" and 'publisher' in config_defaults:
            args.publisher = config_defaults['publisher']
        
        if args.offer == "ubuntu-hpc" and 'offer' in config_defaults:
            args.offer = config_defaults['offer']
        
        if args.sku == "batch.node.ubuntu 22.04" and 'sku' in config_defaults:
            args.sku = config_defaults['sku']
    
    # Auto-generate pool_id and job_id if not provided (use same timestamp for consistency)
    if not args.pool_id or not args.job_id:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        if not args.pool_id:
            args.pool_id = f"mussel-pool-{timestamp}"
            print(f"Auto-generated pool ID: {args.pool_id}")
        
        if not args.job_id:
            args.job_id = f"mussel-job-{timestamp}"
            print(f"Auto-generated job ID: {args.job_id}")

    # Pre-download models if requested and using batch processing
    model_paths = {}
    if args.pre_download_models and pre_download_models and (args.csv_manifest or args.config_file):
        print("\n[Model Pre-Download] Starting model pre-download process...")
        
        # Determine which models need to be downloaded
        models_to_download = []
        
        # Check if user provided explicit model paths (skip pre-download for those)
        # Check both command-line args and config file
        user_provided_paths = {
            'prefilter': args.prefilter_model_path or config_defaults.get('prefilter_model_path'),
            'postfilter': args.postfilter_model_path or config_defaults.get('postfilter_model_path'),
            'slide': args.slide_model_path or config_defaults.get('slide_model_path'),
        }
        
        # Add prefilter model if not provided by user
        if not user_provided_paths['prefilter']:
            # Get the actual prefilter model type from args or config (default: CTRANSPATH)
            prefilter_model_type = getattr(args, 'prefilter_model_type', None) or config_defaults.get('prefilter_model_type', 'CTRANSPATH')
            models_to_download.append(prefilter_model_type)
        
        # Add postfilter models if not provided by user
        if not user_provided_paths['postfilter']:
            # Check for both single postfilter_model_type and multiple postfilter_models
            postfilter_model_type = getattr(args, 'postfilter_model_type', None) or config_defaults.get('postfilter_model_type')
            postfilter_models_arg = args.postfilter_models or config_defaults.get('postfilter_model_types')
            
            if postfilter_model_type:
                # Single postfilter model specified
                models_to_download.append(postfilter_model_type)
            elif postfilter_models_arg:
                # Multiple postfilter models specified (comma-separated)
                postfilter_list = [m.strip() for m in postfilter_models_arg.split(',')]
                models_to_download.extend(postfilter_list)
        
        # Add slide model if not provided by user and slide model type is specified
        slide_model_type = getattr(args, 'slide_model_type', None) or config_defaults.get('slide_model_type')
        if not user_provided_paths['slide'] and slide_model_type:
            models_to_download.append(slide_model_type)
        
        # Remove duplicates
        models_to_download = list(set(models_to_download))
        
        if models_to_download:
            print(f"[Model Pre-Download] Models to download: {', '.join(models_to_download)}")
            
            try:
                # Download models to cache directory
                cached_models = pre_download_models(
                    model_types=models_to_download,
                    cache_dir=args.model_cache_dir
                )
                
                # Upload to S3 if output prefix specified
                if args.output_s3_prefix and upload_models_to_s3:
                    # Determine S3 model prefix
                    if args.model_s3_prefix:
                        s3_model_prefix = args.model_s3_prefix
                    else:
                        # Use output prefix + /models/
                        s3_model_prefix = args.output_s3_prefix.rstrip('/') + '/models/'
                    
                    # Upload models to S3
                    s3_model_paths = upload_models_to_s3(cached_models, s3_model_prefix)
                    model_paths = s3_model_paths
                    print(f"[Model Pre-Download] Models available at S3: {s3_model_prefix}")
                else:
                    # Use local paths (for non-S3 workflows or testing)
                    model_paths = cached_models
                    print(f"[Model Pre-Download] Models cached locally: {args.model_cache_dir}")
                
            except Exception as e:
                print(f"ERROR: Model pre-download failed: {e}", file=sys.stderr)
                print("Continuing with job submission (tasks will download models from HuggingFace Hub)")
        else:
            print("[Model Pre-Download] All models provided by user, skipping pre-download")
    
    # Apply user-provided model paths (override pre-downloaded if both specified)
    if args.prefilter_model_path:
        # Use the actual prefilter model type, not hardcoded CTRANSPATH
        prefilter_model_type = getattr(args, 'prefilter_model_type', None) or config_defaults.get('prefilter_model_type', 'CTRANSPATH')
        model_paths[prefilter_model_type] = args.prefilter_model_path
    if args.postfilter_model_path:
        # Apply to all postfilter models if multiple
        if args.postfilter_models:
            for model in args.postfilter_models.split(','):
                model_paths[model.strip()] = args.postfilter_model_path
    if args.slide_model_path:
        model_paths['slide'] = args.slide_model_path
    
    # Validate CTRANSPATH configuration
    # CTRANSPATH requires a model_path to be provided via configuration
    if args.csv_manifest or args.config_file:
        # Determine the prefilter model type from config or default
        # config_defaults is already loaded above
        prefilter_model = config_defaults.get('prefilter_model_type', 'CTRANSPATH')
        
        # Check if CTRANSPATH is being used without a model_path
        # Check command-line, config file, and pre-downloaded paths
        has_prefilter_path = (
            args.prefilter_model_path or 
            config_defaults.get('prefilter_model_path') or 
            model_paths.get('CTRANSPATH')
        )
        
        if prefilter_model.upper() == 'CTRANSPATH' and not has_prefilter_path:
            print("\n⚠️  WARNING: CTRANSPATH model requires a model_path to be provided via configuration")
            print("   CTRANSPATH does not have a default HuggingFace path and cannot be automatically downloaded.")
            print("   Please provide the model path using one of the following methods:")
            print("     1. Command line: --prefilter-model-path /path/to/ctranspath.pth")
            print("     2. Configuration file: prefilter_model_path: /path/to/ctranspath.pth")
            print("   Tasks will fail if CTRANSPATH model path is not provided.\n")

    # Initialize submitter
    submitter = AzureBatchJobSubmitter(
        batch_account_name=args.batch_account_name,
        batch_account_key=args.batch_account_key,
        batch_account_url=args.batch_account_url,
        storage_account_name=args.storage_account_name,
        storage_account_key=args.storage_account_key,
        azure_files_share_name=args.azure_files_share_name,
    )

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
            default_params['aws_access_key_id'] = args.aws_access_key_id
        if args.aws_secret_access_key:
            default_params['aws_secret_access_key'] = args.aws_secret_access_key
        if args.aws_region:
            default_params['aws_region'] = args.aws_region
        if args.hf_token:
            default_params['hf_token'] = args.hf_token
        if args.max_retry_count is not None:
            default_params['max_retry_count'] = args.max_retry_count
        
        # Add model paths from pre-download or user-provided
        # Command-line args override config and pre-download
        if model_paths.get('CTRANSPATH'):
            default_params['prefilter_model_path'] = model_paths['CTRANSPATH']
        if args.prefilter_model_path:
            default_params['prefilter_model_path'] = args.prefilter_model_path
        # For postfilter, we'll pass the path that applies to all models
        # The postfilter_model_path will be used for all models in the list
        if args.postfilter_models and model_paths:
            # Use the first postfilter model's path if available
            first_model = args.postfilter_models.split(',')[0].strip()
            if first_model in model_paths:
                default_params['postfilter_model_path'] = model_paths[first_model]
        if args.postfilter_model_path:
            default_params['postfilter_model_path'] = args.postfilter_model_path
        if model_paths.get('slide'):
            default_params['slide_model_path'] = model_paths['slide']
        if args.slide_model_path:
            default_params['slide_model_path'] = args.slide_model_path
        
        # Parse postfilter models if provided
        postfilter_models_list = None
        if args.postfilter_models:
            postfilter_models_list = [m.strip() for m in args.postfilter_models.split(',')]
        
        # Use incremental staging and submission if Azure Files staging is enabled
        if args.stage_to_azure_files:
            if not args.azure_files_share_name:
                print("ERROR: --azure-files-share-name required when using --stage-to-azure-files")
                sys.exit(1)
            
            # Stage and submit tasks incrementally
            submitter.stage_and_submit_tasks_from_csv(
                job_id=args.job_id,
                csv_file=args.csv_manifest,
                output_dir=args.output_dir,
                output_s3_prefix=args.output_s3_prefix,
                container_image=args.container_image,
                postfilter_models=postfilter_models_list,
                remote_dir="slides",
                **default_params,
            )
        else:
            # Standard workflow without staging
            submitter.submit_tasks_from_csv(
                job_id=args.job_id,
                csv_file=args.csv_manifest,
                output_dir=args.output_dir,
                output_s3_prefix=args.output_s3_prefix,
                container_image=args.container_image,
                postfilter_models=postfilter_models_list,
                distributed_slide_batch_size=args.distributed_slide_batch_size,
                **default_params,
            )
    elif args.task_id and args.slide_path:
        # Single task submission
        # Prepare model paths for single task
        task_model_paths = {}
        if model_paths.get('CTRANSPATH'):
            task_model_paths['prefilter_model_path'] = model_paths['CTRANSPATH']
        if model_paths:
            # Try to find a postfilter model path
            for key, path in model_paths.items():
                if key != 'CTRANSPATH' and key != 'slide':
                    task_model_paths['postfilter_model_path'] = path
                    break
        if model_paths.get('slide'):
            task_model_paths['slide_model_path'] = model_paths['slide']
        
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
        print("ERROR: Must specify either --csv-manifest or --task-id with --slide-path")
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
