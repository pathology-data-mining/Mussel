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

    def __init__(
        self,
        batch_account_name: str,
        batch_account_key: str,
        batch_account_url: str,
        storage_account_name: Optional[str] = None,
        storage_account_key: Optional[str] = None,
    ):
        """Initialize Azure Batch client."""
        credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
        self.batch_client = BatchServiceClient(credentials, batch_account_url)
        
        self.storage_account_name = storage_account_name
        self.storage_account_key = storage_account_key
        
        # Track task metadata for failure reporting
        self.task_metadata = {}
        
        if storage_account_name and storage_account_key:
            self.blob_client = BlobServiceClient(
                account_url=f"https://{storage_account_name}.blob.core.windows.net",
                credential=storage_account_key,
            )
        else:
            self.blob_client = None

    def create_pool(
        self,
        pool_id: str,
        vm_size: str = "Standard_NC6s_v3",
        node_count: int = 1,
        container_image: str = "mskmind/mussel:latest-torch-gpu",
        use_gpu: bool = True,
    ) -> None:
        """Create a pool of compute nodes."""
        print(f"Creating pool '{pool_id}'...")

        # Container configuration
        container_conf = batchmodels.ContainerConfiguration(
            container_image_names=[container_image]
        )

        # VM configuration
        image_ref = batchmodels.ImageReference(
            publisher="microsoft-azure-batch",
            offer="ubuntu-server-container",
            sku="20-04-lts",
            version="latest",
        )

        vm_config = batchmodels.VirtualMachineConfiguration(
            image_reference=image_ref,
            container_configuration=container_conf,
            node_agent_sku_id="batch.node.ubuntu 20.04",
        )

        # Pool configuration
        pool = batchmodels.PoolAddParameter(
            id=pool_id,
            virtual_machine_configuration=vm_config,
            vm_size=vm_size,
            target_dedicated_nodes=node_count,
            enable_auto_scale=False,
        )

        try:
            self.batch_client.pool.add(pool)
            print(f"Pool '{pool_id}' created successfully")
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
        postfilter_model_type: Optional[str] = None,
        segment_threshold: int = 0,
        patch_size: int = 256,
        mpp: float = 0.5,
        num_workers: int = 4,
        batch_size: int = 64,
        use_gpu: bool = True,
        keep_intermediate_files: bool = False,
        hf_token: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        max_retry_count: int = 3,
        container_image: str = "mskmind/mussel:latest-torch-gpu",
    ) -> None:
        """Submit a tessellate-extract-features task to Azure Batch."""
        print(f"Submitting task '{task_id}' to job '{job_id}'...")

        # Build environment variables
        env_vars = [
            batchmodels.EnvironmentSetting("SLIDE_PATH", slide_path),
            batchmodels.EnvironmentSetting("OUTPUT_H5_PATH", output_h5_path),
            batchmodels.EnvironmentSetting("OUTPUT_PT_PATH", output_pt_path),
            batchmodels.EnvironmentSetting("PREFILTER_MODEL_TYPE", prefilter_model_type),
            batchmodels.EnvironmentSetting("SEGMENT_THRESHOLD", str(segment_threshold)),
            batchmodels.EnvironmentSetting("PATCH_SIZE", str(patch_size)),
            batchmodels.EnvironmentSetting("MPP", str(mpp)),
            batchmodels.EnvironmentSetting("NUM_WORKERS", str(num_workers)),
            batchmodels.EnvironmentSetting("BATCH_SIZE", str(batch_size)),
            batchmodels.EnvironmentSetting("USE_GPU", str(use_gpu).lower()),
            batchmodels.EnvironmentSetting("KEEP_INTERMEDIATE_FILES", str(keep_intermediate_files).lower()),
            batchmodels.EnvironmentSetting("AGGREGATION_METHOD", aggregation_method),
        ]

        if intermediate_h5_path:
            env_vars.append(batchmodels.EnvironmentSetting("INTERMEDIATE_H5_PATH", intermediate_h5_path))

        if slide_model_type:
            env_vars.append(batchmodels.EnvironmentSetting("SLIDE_MODEL_TYPE", slide_model_type))

        if classifier_pkl:
            env_vars.append(batchmodels.EnvironmentSetting("CLASSIFIER_PKL", classifier_pkl))
            env_vars.append(batchmodels.EnvironmentSetting("CLASSIFIER_THRESHOLD", str(classifier_threshold)))

        if postfilter_model_type:
            env_vars.append(batchmodels.EnvironmentSetting("POSTFILTER_MODEL_TYPE", postfilter_model_type))

        if hf_token:
            env_vars.append(batchmodels.EnvironmentSetting("HF_TOKEN", hf_token))

        # Add AWS credentials if provided (for S3 access)
        if aws_access_key_id:
            env_vars.append(batchmodels.EnvironmentSetting("AWS_ACCESS_KEY_ID", aws_access_key_id))
        if aws_secret_access_key:
            env_vars.append(batchmodels.EnvironmentSetting("AWS_SECRET_ACCESS_KEY", aws_secret_access_key))
        if aws_region:
            env_vars.append(batchmodels.EnvironmentSetting("AWS_DEFAULT_REGION", aws_region))

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

    def submit_tasks_from_config(
        self,
        job_id: str,
        config_file: str,
        container_image: str = "mskmind/mussel:latest-torch-gpu",
    ) -> None:
        """Submit multiple tasks from a configuration file."""
        print(f"Loading task configuration from '{config_file}'...")

        with open(config_file, 'r') as f:
            config = json.load(f)

        tasks = config.get("tasks", [])
        defaults = config.get("defaults", {})

        print(f"Submitting {len(tasks)} tasks...")

        for i, task_config in enumerate(tasks):
            # Merge with defaults
            merged_config = {**defaults, **task_config}

            task_id = merged_config.get("task_id", f"task_{i}")

            self.submit_task(
                job_id=job_id,
                task_id=task_id,
                slide_path=merged_config["slide_path"],
                output_h5_path=merged_config["output_h5_path"],
                output_pt_path=merged_config["output_pt_path"],
                intermediate_h5_path=merged_config.get("intermediate_h5_path"),
                aggregation_method=merged_config.get("aggregation_method", "identity"),
                slide_model_type=merged_config.get("slide_model_type"),
                classifier_pkl=merged_config.get("classifier_pkl"),
                classifier_threshold=merged_config.get("classifier_threshold", 0.75),
                prefilter_model_type=merged_config.get("prefilter_model_type", "CTRANSPATH"),
                postfilter_model_type=merged_config.get("postfilter_model_type"),
                segment_threshold=merged_config.get("segment_threshold", 0),
                patch_size=merged_config.get("patch_size", 256),
                mpp=merged_config.get("mpp", 0.5),
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

    def submit_tasks_from_csv(
        self,
        job_id: str,
        csv_file: str,
        output_dir: str = "/mnt/output",
        output_s3_prefix: Optional[str] = None,
        container_image: str = "mskmind/mussel:latest-torch-gpu",
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
            **default_params: Default parameters for all tasks (e.g., prefilter_model_type, batch_size)
        """
        print(f"Loading task manifest from '{csv_file}'...")

        tasks = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                slide_id = row['slide_id']
                slide_path = row['slide_path']
                
                # Determine output paths
                if output_s3_prefix:
                    # Upload to S3 (slide-level features)
                    output_h5_path = f"{output_s3_prefix.rstrip('/')}/{slide_id}_features.h5"
                    output_pt_path = f"{output_s3_prefix.rstrip('/')}/{slide_id}_features.pt"
                    # Tile-level features (when doing aggregation)
                    intermediate_h5_path = f"{output_s3_prefix.rstrip('/')}/{slide_id}_tile_features.h5"
                else:
                    # Local output (slide-level features)
                    output_h5_path = f"{output_dir}/{slide_id}_features.h5"
                    output_pt_path = f"{output_dir}/{slide_id}_features.pt"
                    # Tile-level features (when doing aggregation)
                    intermediate_h5_path = f"{output_dir}/{slide_id}_tile_features.h5"
                
                tasks.append({
                    'task_id': slide_id,
                    'slide_path': slide_path,
                    'output_h5_path': output_h5_path,
                    'output_pt_path': output_pt_path,
                    'intermediate_h5_path': intermediate_h5_path,
                })

        print(f"Submitting {len(tasks)} tasks from CSV manifest...")

        for task_config in tasks:
            # Merge with default parameters
            merged_config = {**default_params, **task_config}

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
                segment_threshold=merged_config.get("segment_threshold", 0),
                patch_size=merged_config.get("patch_size", 256),
                mpp=merged_config.get("mpp", 0.5),
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
    
    # Azure credentials
    parser.add_argument("--batch-account-name", required=True, help="Azure Batch account name")
    parser.add_argument("--batch-account-key", required=True, help="Azure Batch account key")
    parser.add_argument("--batch-account-url", required=True, help="Azure Batch account URL")
    parser.add_argument("--storage-account-name", help="Azure Storage account name (optional)")
    parser.add_argument("--storage-account-key", help="Azure Storage account key (optional)")
    
    # Pool configuration
    parser.add_argument("--pool-id", required=True, help="Pool ID")
    parser.add_argument("--create-pool", action="store_true", help="Create pool if it doesn't exist")
    parser.add_argument("--vm-size", default="Standard_NC6s_v3", help="VM size for pool nodes")
    parser.add_argument("--node-count", type=int, default=1, help="Number of nodes in pool")
    parser.add_argument("--container-image", default="mskmind/mussel:latest-torch-gpu", 
                        help="Docker container image")
    
    # Job configuration
    parser.add_argument("--job-id", required=True, help="Job ID")
    parser.add_argument("--create-job", action="store_true", help="Create job")
    
    # Task configuration
    parser.add_argument("--config-file", help="JSON file with task configurations")
    parser.add_argument("--csv-manifest", help="CSV manifest file with slide_id,slide_path columns")
    parser.add_argument("--output-dir", default="/mnt/output", help="Output directory for results (when using CSV)")
    parser.add_argument("--output-s3-prefix", help="S3 prefix for outputs (e.g., s3://bucket/results/)")
    parser.add_argument("--task-id", help="Single task ID")
    parser.add_argument("--slide-path", help="Path to slide file (for single task, can be s3://)")
    parser.add_argument("--output-h5-path", help="Output H5 path (for single task, can be s3://)")
    parser.add_argument("--output-pt-path", help="Output PT path (for single task, can be s3://)")
    
    # AWS credentials for S3 access
    parser.add_argument("--aws-access-key-id", help="AWS access key ID for S3")
    parser.add_argument("--aws-secret-access-key", help="AWS secret access key for S3")
    parser.add_argument("--aws-region", default="us-east-1", help="AWS region")
    
    # Retry configuration
    parser.add_argument("--max-retry-count", type=int, default=3, help="Maximum number of retry attempts for failed tasks (default: 3)")
    parser.add_argument("--save-failed-tasks", help="Save failed tasks to CSV file for resubmission")
    
    # Monitoring and cleanup
    parser.add_argument("--monitor", action="store_true", help="Monitor task progress")
    parser.add_argument("--delete-job", action="store_true", help="Delete job after completion")
    parser.add_argument("--delete-pool", action="store_true", help="Delete pool after completion")
    
    args = parser.parse_args()

    # Initialize submitter
    submitter = AzureBatchJobSubmitter(
        batch_account_name=args.batch_account_name,
        batch_account_key=args.batch_account_key,
        batch_account_url=args.batch_account_url,
        storage_account_name=args.storage_account_name,
        storage_account_key=args.storage_account_key,
    )

    # Create pool if requested
    if args.create_pool:
        submitter.create_pool(
            pool_id=args.pool_id,
            vm_size=args.vm_size,
            node_count=args.node_count,
            container_image=args.container_image,
        )

    # Create job if requested
    if args.create_job:
        submitter.create_job(job_id=args.job_id, pool_id=args.pool_id)

    # Submit tasks
    if args.config_file:
        submitter.submit_tasks_from_config(
            job_id=args.job_id,
            config_file=args.config_file,
            container_image=args.container_image,
        )
    elif args.csv_manifest:
        # Prepare default parameters for CSV tasks
        default_params = {}
        if args.aws_access_key_id:
            default_params['aws_access_key_id'] = args.aws_access_key_id
        if args.aws_secret_access_key:
            default_params['aws_secret_access_key'] = args.aws_secret_access_key
        if args.aws_region:
            default_params['aws_region'] = args.aws_region
        if args.max_retry_count is not None:
            default_params['max_retry_count'] = args.max_retry_count
        
        submitter.submit_tasks_from_csv(
            job_id=args.job_id,
            csv_file=args.csv_manifest,
            output_dir=args.output_dir,
            output_s3_prefix=args.output_s3_prefix,
            container_image=args.container_image,
            **default_params,
        )
    elif args.task_id and args.slide_path:
        submitter.submit_task(
            job_id=args.job_id,
            task_id=args.task_id,
            slide_path=args.slide_path,
            output_h5_path=args.output_h5_path,
            output_pt_path=args.output_pt_path,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            aws_region=args.aws_region,
            max_retry_count=args.max_retry_count,
            container_image=args.container_image,
        )

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

    # Cleanup if requested
    if args.delete_job:
        submitter.delete_job(job_id=args.job_id)
    
    if args.delete_pool:
        submitter.delete_pool(pool_id=args.pool_id)


if __name__ == "__main__":
    main()
