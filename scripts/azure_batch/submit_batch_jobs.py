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
        ]

        if classifier_pkl:
            env_vars.append(batchmodels.EnvironmentSetting("CLASSIFIER_PKL", classifier_pkl))
            env_vars.append(batchmodels.EnvironmentSetting("CLASSIFIER_THRESHOLD", str(classifier_threshold)))

        if postfilter_model_type:
            env_vars.append(batchmodels.EnvironmentSetting("POSTFILTER_MODEL_TYPE", postfilter_model_type))

        if hf_token:
            env_vars.append(batchmodels.EnvironmentSetting("HF_TOKEN", hf_token))

        # Container settings
        container_settings = batchmodels.TaskContainerSettings(
            image_name=container_image,
            container_run_options="--rm --gpus all" if use_gpu else "--rm",
        )

        # Task command - run the task script
        task_command = "/bin/bash /app/scripts/azure_batch/run_tessellate_extract_features.sh"

        # Create task
        task = batchmodels.TaskAddParameter(
            id=task_id,
            command_line=task_command,
            container_settings=container_settings,
            environment_settings=env_vars,
        )

        try:
            self.batch_client.task.add(job_id, task)
            print(f"Task '{task_id}' submitted successfully")
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
    parser.add_argument("--task-id", help="Single task ID")
    parser.add_argument("--slide-path", help="Path to slide file (for single task)")
    parser.add_argument("--output-h5-path", help="Output H5 path (for single task)")
    parser.add_argument("--output-pt-path", help="Output PT path (for single task)")
    
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
    elif args.task_id and args.slide_path:
        submitter.submit_task(
            job_id=args.job_id,
            task_id=args.task_id,
            slide_path=args.slide_path,
            output_h5_path=args.output_h5_path,
            output_pt_path=args.output_pt_path,
            container_image=args.container_image,
        )

    # Monitor if requested
    if args.monitor:
        submitter.monitor_tasks(job_id=args.job_id)

    # Cleanup if requested
    if args.delete_job:
        submitter.delete_job(job_id=args.job_id)
    
    if args.delete_pool:
        submitter.delete_pool(pool_id=args.pool_id)


if __name__ == "__main__":
    main()
