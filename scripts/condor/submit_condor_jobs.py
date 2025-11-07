#!/usr/bin/env python3
"""
HTCondor job submission script for tessellate-extract-features.

This script submits one or more tessellate-extract-features tasks to HTCondor.
It handles submit file generation, job submission, and monitoring.

Requirements:
    - HTCondor installed and configured
    - Access to HTCondor submit node

Install HTCondor Python bindings (optional): pip install htcondor
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Import model pre-download utility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
try:
    from model_predownload import pre_download_models
except ImportError:
    print("WARNING: Could not import model_predownload module. Pre-download features will be unavailable.")
    pre_download_models = None


class CondorJobSubmitter:
    """
    Handles HTCondor job submission for Mussel tessellate-extract-features.
    
    This class provides methods to:
    - Generate HTCondor submit files
    - Submit single tasks or batch tasks from CSV manifests
    - Monitor job progress
    - Handle retries via DAGMan
    
    Typical usage:
        submitter = CondorJobSubmitter()
        submitter.submit_tasks_from_csv("manifest.csv", output_dir="/output")
    """

    def __init__(self, script_dir: Optional[str] = None):
        """Initialize HTCondor job submitter."""
        self.script_dir = script_dir or str(Path(__file__).parent)
        self.task_script = str(Path(__file__).parent.parent / "common" / "run_tessellate_extract_features.sh")
        
        # Check if task script exists
        if not os.path.exists(self.task_script):
            print(f"ERROR: Task script not found: {self.task_script}")
            sys.exit(1)

    def generate_submit_file(
        self,
        task_id: str,
        slide_path: str = None,
        slide_paths: Optional[List[str]] = None,
        slide_ids: Optional[List[str]] = None,
        output_h5_path: str = None,
        output_pt_path: str = None,
        output_dir_for_batch: Optional[str] = None,
        intermediate_h5_path: Optional[str] = None,
        classifier_pkl: Optional[str] = None,
        classifier_threshold: float = 0.75,
        prefilter_model_type: str = "CTRANSPATH",
        prefilter_model_path: Optional[str] = None,
        postfilter_model_type: Optional[str] = None,
        postfilter_model_path: Optional[str] = None,
        postfilter_model_types: Optional[str] = None,
        aggregation_method: Optional[str] = None,
        slide_model_type: Optional[str] = None,
        slide_model_path: Optional[str] = None,
        segment_threshold: int = 0,
        patch_size: int = 256,
        mpp: float = 0.5,
        num_workers: int = 4,
        batch_size: int = 64,
        use_gpu: bool = True,
        request_cpus: int = 4,
        request_memory: str = "16GB",
        request_gpus: int = 1 if True else 0,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = "us-east-1",
        aws_endpoint_url: Optional[str] = None,
        hf_token: Optional[str] = None,
        max_retries: int = 3,
        output_dir: Optional[str] = None,
        slide_batch_size: int = 8,
    ) -> str:
        """Generate HTCondor submit file content.
        
        Supports both single-slide and multi-slide batch processing.
        For batch processing, provide slide_paths and slide_ids instead of slide_path.
        """
        
        # Set output directory for logs
        log_dir = output_dir or "condor_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Build environment variables
        env_vars = {}
        
        # Handle batch vs single slide processing
        if slide_paths and len(slide_paths) > 1:
            # Batch processing mode
            env_vars["SLIDE_PATHS"] = ",".join(slide_paths)
            if slide_ids:
                env_vars["SLIDE_IDS"] = ",".join(slide_ids)
            if output_dir_for_batch:
                env_vars["OUTPUT_DIR"] = output_dir_for_batch
            env_vars["SLIDE_BATCH_SIZE"] = str(slide_batch_size)
        else:
            # Single slide mode (backward compatible)
            # If slide_paths has one element, use it; otherwise use slide_path parameter
            if slide_paths and not slide_path:
                slide_path = slide_paths[0]
            env_vars["SLIDE_PATH"] = slide_path
            env_vars["OUTPUT_H5_PATH"] = output_h5_path
            env_vars["OUTPUT_PT_PATH"] = output_pt_path
        
        # Common environment variables
        env_vars.update({
            "CLASSIFIER_THRESHOLD": str(classifier_threshold),
            "PREFILTER_MODEL_TYPE": prefilter_model_type,
            "SEGMENT_THRESHOLD": str(segment_threshold),
            "PATCH_SIZE": str(patch_size),
            "MPP": str(mpp),
            "NUM_WORKERS": str(num_workers),
            "BATCH_SIZE": str(batch_size),
            "USE_GPU": "true" if use_gpu else "false",
            "AWS_DEFAULT_REGION": aws_region,
        })
        
        if intermediate_h5_path:
            env_vars["INTERMEDIATE_H5_PATH"] = intermediate_h5_path
        if classifier_pkl:
            env_vars["CLASSIFIER_PKL"] = classifier_pkl
        if prefilter_model_path:
            env_vars["PREFILTER_MODEL_PATH"] = prefilter_model_path
        if postfilter_model_type:
            env_vars["POSTFILTER_MODEL_TYPE"] = postfilter_model_type
        if postfilter_model_path:
            env_vars["POSTFILTER_MODEL_PATH"] = postfilter_model_path
        if postfilter_model_types:
            env_vars["POSTFILTER_MODEL_TYPES"] = postfilter_model_types
        if aggregation_method:
            env_vars["AGGREGATION_METHOD"] = aggregation_method
        if slide_model_type:
            env_vars["SLIDE_MODEL_TYPE"] = slide_model_type
        if slide_model_path:
            env_vars["SLIDE_MODEL_PATH"] = slide_model_path
        if aws_access_key_id:
            env_vars["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        if aws_secret_access_key:
            env_vars["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        if aws_endpoint_url:
            env_vars["AWS_ENDPOINT_URL"] = aws_endpoint_url
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token
        
        # Format environment string for HTCondor
        env_string = " ".join([f"{k}={v}" for k, v in env_vars.items()])
        
        # GPU requirements
        gpu_require = ""
        if use_gpu and request_gpus > 0:
            gpu_require = f"request_gpus = {request_gpus}\n"
            gpu_require += "requirements = (CUDACapability >= 3.5)\n"
        
        # Generate submit file
        submit_content = f"""# HTCondor submit file for {task_id}
universe = vanilla
executable = {self.task_script}

# Resource requirements
request_cpus = {request_cpus}
request_memory = {request_memory}
{gpu_require}
# Environment variables
environment = "{env_string}"

# Output/error/log files
output = {log_dir}/{task_id}.out
error = {log_dir}/{task_id}.err
log = {log_dir}/{task_id}.log

# Retry configuration
max_retries = {max_retries}

# Transfer files
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

# Submit
queue 1
"""
        
        return submit_content

    def submit_task(
        self,
        task_id: str,
        slide_path: str,
        output_h5_path: str,
        output_pt_path: str,
        **kwargs
    ) -> Optional[str]:
        """
        Submit a single task to HTCondor.
        
        Returns:
            Job ID if successful, None otherwise
        """
        print(f"Submitting task: {task_id}")
        
        # Generate submit file
        submit_content = self.generate_submit_file(
            task_id=task_id,
            slide_path=slide_path,
            output_h5_path=output_h5_path,
            output_pt_path=output_pt_path,
            **kwargs
        )
        
        # Write submit file
        submit_file = f"condor_submit_{task_id}.sub"
        with open(submit_file, 'w') as f:
            f.write(submit_content)
        
        print(f"Generated submit file: {submit_file}")
        
        # Submit to HTCondor
        if kwargs.get('submit', False):
            try:
                result = subprocess.run(
                    ["condor_submit", submit_file],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(result.stdout)
                # Extract job ID from output
                for line in result.stdout.split('\n'):
                    if 'submitted to cluster' in line.lower():
                        job_id = line.split()[-1].rstrip('.')
                        print(f"Job ID: {job_id}")
                        return job_id
            except subprocess.CalledProcessError as e:
                print(f"ERROR submitting job: {e}")
                print(e.stderr)
                return None
        else:
            print("Dry run - submit file generated but not submitted")
            print("Use --submit to actually submit to HTCondor")
        
        return None

    def _should_use_batch_encoding(self, **kwargs) -> bool:
        """Determine if we should use batch slide encoding optimization.
        
        Batch encoding is beneficial when:
        1. Using model-based aggregation (aggregation_method="model")
        2. Using a slide encoder (slide_model_type is specified)
        3. Processing multiple slides
        """
        return (
            kwargs.get('aggregation_method') == 'model' and
            kwargs.get('slide_model_type') is not None
        )

    def submit_tasks_from_csv(
        self,
        csv_file: str,
        output_dir: Optional[str] = None,
        output_s3_prefix: Optional[str] = None,
        distributed_slide_batch_size: int = 1,
        **kwargs
    ) -> List[Optional[str]]:
        """
        Submit multiple tasks from a CSV manifest.
        
        CSV format: slide_id,slide_path
        
        Args:
            csv_file: Path to CSV manifest file
            output_dir: Output directory for results
            output_s3_prefix: S3 prefix for outputs
            distributed_slide_batch_size: Number of slides to group per task for batch encoding (default: 1).
                When > 1 and using slide-level model aggregation, slides are grouped into batches
                to optimize slide encoder loading. Recommended: 8-16 for GIGAPATH_SLIDE/TITAN_SLIDE.
            **kwargs: Additional task parameters
        
        Returns:
            List of job IDs
        """
        print(f"Reading CSV manifest: {csv_file}")
        
        # Read all slides from CSV
        slides = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                slides.append({
                    'slide_id': row['slide_id'],
                    'slide_path': row['slide_path']
                })
        
        # Determine if we should use batch encoding
        use_batch_encoding = (
            distributed_slide_batch_size > 1 and 
            self._should_use_batch_encoding(**kwargs)
        )
        
        if use_batch_encoding:
            print(f"\n[Batch Encoding Optimization] Enabled")
            print(f"  Grouping slides into batches of {distributed_slide_batch_size}")
            print(f"  Slide encoder: {kwargs.get('slide_model_type')}")
            print(f"  This reduces model loading overhead from {len(slides)}x to {(len(slides) + distributed_slide_batch_size - 1) // distributed_slide_batch_size}x")
        
        job_ids = []
        
        # Process slides in batches if batch encoding is enabled
        if use_batch_encoding:
            # Group slides into batches
            for batch_idx in range(0, len(slides), distributed_slide_batch_size):
                batch_slides = slides[batch_idx:batch_idx + distributed_slide_batch_size]
                batch_size = len(batch_slides)
                
                # Create batch task ID
                batch_id = f"batch_{batch_idx // distributed_slide_batch_size + 1}_of_{(len(slides) + distributed_slide_batch_size - 1) // distributed_slide_batch_size}"
                
                # Extract slide IDs and paths for this batch
                slide_ids = [s['slide_id'] for s in batch_slides]
                slide_paths = [s['slide_path'] for s in batch_slides]
                
                print(f"\nSubmitting batch task: {batch_id}")
                print(f"  Slides: {', '.join(slide_ids)}")
                
                # Determine output directory for batch
                if output_s3_prefix:
                    model_type = kwargs.get('prefilter_model_type', 'CTRANSPATH')
                    if kwargs.get('postfilter_model_types'):
                        models = kwargs['postfilter_model_types'].split(',')
                        model_type = models[0]
                    output_dir_for_batch = f"{output_s3_prefix.rstrip('/')}/{model_type}"
                elif output_dir:
                    output_dir_for_batch = output_dir
                else:
                    print(f"ERROR: Must specify --output-dir or --output-s3-prefix")
                    sys.exit(1)
                
                # Submit batch task
                job_id = self.submit_task(
                    task_id=batch_id,
                    slide_paths=slide_paths,
                    slide_ids=slide_ids,
                    output_dir_for_batch=output_dir_for_batch,
                    output_dir=kwargs.get('output_dir', 'condor_logs'),
                    slide_batch_size=kwargs.get('slide_batch_size', 8),
                    **kwargs
                )
                job_ids.append(job_id)
        else:
            # Single-slide tasks (original behavior)
            for slide in slides:
                slide_id = slide['slide_id']
                slide_path = slide['slide_path']
                
                # Generate output paths
                if output_s3_prefix:
                    prefix = output_s3_prefix.rstrip('/')
                    model_type = kwargs.get('prefilter_model_type', 'CTRANSPATH')
                    
                    # Check if multi-model mode
                    if kwargs.get('postfilter_model_types'):
                        # Multi-model: organize by model type
                        models = kwargs['postfilter_model_types'].split(',')
                        model_type = models[0]  # Use first model for primary outputs
                    
                    output_h5_path = f"{prefix}/{model_type}/h5/{slide_id}_features.h5"
                    output_pt_path = f"{prefix}/{model_type}/pt/{slide_id}_features.pt"
                    
                    if kwargs.get('aggregation_method'):
                        intermediate_h5_path = f"{prefix}/{model_type}/tile_h5/{slide_id}_tile_features.h5"
                        kwargs['intermediate_h5_path'] = intermediate_h5_path
                
                elif output_dir:
                    output_h5_path = os.path.join(output_dir, f"{slide_id}_features.h5")
                    output_pt_path = os.path.join(output_dir, f"{slide_id}_features.pt")
                else:
                    print(f"ERROR: Must specify --output-dir or --output-s3-prefix")
                    sys.exit(1)
                
                # Submit task
                job_id = self.submit_task(
                    task_id=slide_id,
                    slide_path=slide_path,
                    output_h5_path=output_h5_path,
                    output_pt_path=output_pt_path,
                    output_dir=kwargs.get('output_dir', 'condor_logs'),
                    **kwargs
                )
                job_ids.append(job_id)
        
        print(f"\nSubmitted {len(job_ids)} tasks")
        return job_ids


def main():
    parser = argparse.ArgumentParser(
        description="Submit tessellate-extract-features jobs to HTCondor"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--task-id", help="Single task ID")
    input_group.add_argument("--csv-manifest", help="CSV manifest file with slide_id,slide_path")
    
    # Slide parameters
    parser.add_argument("--slide-path", help="Path to slide file (required for single task)")
    parser.add_argument("--output-h5-path", help="Output HDF5 file path (required for single task)")
    parser.add_argument("--output-pt-path", help="Output PyTorch file path (required for single task)")
    parser.add_argument("--output-dir", help="Output directory for batch processing")
    parser.add_argument("--output-s3-prefix", help="S3 prefix for organized outputs")
    
    # Processing parameters
    parser.add_argument("--classifier-pkl", help="Classifier pickle file for filtering")
    parser.add_argument("--classifier-threshold", type=float, default=0.75)
    parser.add_argument("--prefilter-model-type", default="CTRANSPATH")
    parser.add_argument("--postfilter-model-type", help="Single postfilter model type")
    parser.add_argument("--postfilter-models", help="Comma-separated list of postfilter models")
    parser.add_argument("--aggregation-method", choices=["identity", "mean", "max", "model"])
    parser.add_argument("--slide-model-type", help="Model for aggregation_method=model")
    parser.add_argument("--segment-threshold", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--mpp", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--slide-batch-size", type=int, default=8, 
                        help="Slides per batch during slide-level aggregation (default: 8)")
    parser.add_argument("--distributed-slide-batch-size", type=int, default=1,
                        help="Number of slides to group per distributed task for batch encoding optimization (default: 1). "
                             "When > 1 and using slide-level model aggregation (e.g., GIGAPATH_SLIDE), slides are grouped "
                             "into batches to optimize slide encoder loading. Recommended: 8-16 for better efficiency.")
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", action="store_false", dest="use_gpu")
    
    # Resource requirements
    parser.add_argument("--request-cpus", type=int, default=4)
    parser.add_argument("--request-memory", default="16GB")
    parser.add_argument("--request-gpus", type=int, default=1)
    
    # AWS S3 credentials
    parser.add_argument("--aws-access-key-id", help="AWS access key ID")
    parser.add_argument("--aws-secret-access-key", help="AWS secret access key")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--aws-endpoint-url", help="Custom S3 endpoint URL (e.g., for MinIO or Ceph)")
    
    # HuggingFace token
    parser.add_argument("--hf-token", help="HuggingFace token for gated models")
    
    # Model pre-download configuration
    parser.add_argument("--pre-download-models", action="store_true", default=True,
                        help="Pre-download models before job submission (default: True for batch jobs)")
    parser.add_argument("--no-pre-download-models", dest="pre_download_models", action="store_false",
                        help="Disable model pre-download")
    parser.add_argument("--model-cache-dir", default="./model_cache",
                        help="Shared filesystem directory to cache models (default: ./model_cache)")
    parser.add_argument("--prefilter-model-path", help="Path to prefilter model weights (local filesystem)")
    parser.add_argument("--postfilter-model-path", help="Path to postfilter model weights (local filesystem)")
    parser.add_argument("--slide-model-path", help="Path to slide encoder model weights (local filesystem)")
    
    # Retry configuration
    parser.add_argument("--max-retries", type=int, default=3)
    
    # Submission
    parser.add_argument("--submit", action="store_true", help="Actually submit to HTCondor")
    
    args = parser.parse_args()
    
    # Pre-download models if requested and using batch processing
    model_paths = {}
    if args.pre_download_models and pre_download_models and args.csv_manifest:
        print("\n[Model Pre-Download] Starting model pre-download process...")
        
        # Determine which models need to be downloaded
        models_to_download = []
        
        # Check if user provided explicit model paths (skip pre-download for those)
        user_provided_paths = {
            'prefilter': args.prefilter_model_path,
            'postfilter': args.postfilter_model_path,
            'slide': args.slide_model_path,
        }
        
        # Add prefilter model if not provided by user
        if not user_provided_paths['prefilter']:
            # Default prefilter is CTRANSPATH
            models_to_download.append('CTRANSPATH')
        
        # Add postfilter models if not provided by user
        if not user_provided_paths['postfilter'] and args.postfilter_models:
            postfilter_list = [m.strip() for m in args.postfilter_models.split(',')]
            models_to_download.extend(postfilter_list)
        
        # Remove duplicates
        models_to_download = list(set(models_to_download))
        
        if models_to_download:
            print(f"[Model Pre-Download] Models to download: {', '.join(models_to_download)}")
            
            try:
                # Download models to shared filesystem cache directory
                cached_models = pre_download_models(
                    model_types=models_to_download,
                    cache_dir=args.model_cache_dir
                )
                
                model_paths = cached_models
                print(f"[Model Pre-Download] Models cached to shared filesystem: {args.model_cache_dir}")
                
            except Exception as e:
                print(f"ERROR: Model pre-download failed: {e}", file=sys.stderr)
                print("Continuing with job submission (tasks will download models from HuggingFace Hub)")
        else:
            print("[Model Pre-Download] All models provided by user, skipping pre-download")
    
    # Apply user-provided model paths (override pre-downloaded if both specified)
    if args.prefilter_model_path:
        model_paths['CTRANSPATH'] = args.prefilter_model_path  # Assume prefilter is CTRANSPATH
    if args.postfilter_model_path:
        # Apply to all postfilter models if multi-model
        if args.postfilter_models:
            for model in args.postfilter_models.split(','):
                model_paths[model.strip()] = args.postfilter_model_path
        elif args.postfilter_model_type:
            model_paths[args.postfilter_model_type] = args.postfilter_model_path
    if args.slide_model_path and args.slide_model_type:
        model_paths[args.slide_model_type] = args.slide_model_path
    
    # Validate single task arguments
    if args.task_id:
        if not args.slide_path or not args.output_h5_path or not args.output_pt_path:
            parser.error("--task-id requires --slide-path, --output-h5-path, and --output-pt-path")
    
    # Create submitter
    submitter = CondorJobSubmitter()
    
    # Submit tasks
    if args.task_id:
        # Determine model paths for single task
        task_model_paths = {}
        if model_paths:
            task_model_paths = {
                'prefilter': model_paths.get('CTRANSPATH', args.prefilter_model_path),
                'postfilter': model_paths.get(args.postfilter_model_type, args.postfilter_model_path) if args.postfilter_model_type else None,
                'slide': model_paths.get(args.slide_model_type, args.slide_model_path) if args.slide_model_type else None,
            }
        
        submitter.submit_task(
            task_id=args.task_id,
            slide_path=args.slide_path,
            output_h5_path=args.output_h5_path,
            output_pt_path=args.output_pt_path,
            classifier_pkl=args.classifier_pkl,
            classifier_threshold=args.classifier_threshold,
            prefilter_model_type=args.prefilter_model_type,
            prefilter_model_path=task_model_paths.get('prefilter'),
            postfilter_model_type=args.postfilter_model_type,
            postfilter_model_path=task_model_paths.get('postfilter'),
            postfilter_model_types=args.postfilter_models,
            aggregation_method=args.aggregation_method,
            slide_model_type=args.slide_model_type,
            slide_model_path=task_model_paths.get('slide'),
            segment_threshold=args.segment_threshold,
            patch_size=args.patch_size,
            mpp=args.mpp,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            use_gpu=args.use_gpu,
            request_cpus=args.request_cpus,
            request_memory=args.request_memory,
            request_gpus=args.request_gpus if args.use_gpu else 0,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            aws_region=args.aws_region,
            hf_token=args.hf_token,
            max_retries=args.max_retries,
            submit=args.submit,
        )
    else:
        submitter.submit_tasks_from_csv(
            csv_file=args.csv_manifest,
            output_dir=args.output_dir,
            output_s3_prefix=args.output_s3_prefix,
            distributed_slide_batch_size=args.distributed_slide_batch_size,
            classifier_pkl=args.classifier_pkl,
            classifier_threshold=args.classifier_threshold,
            prefilter_model_type=args.prefilter_model_type,
            prefilter_model_path=model_paths.get('CTRANSPATH') if model_paths else None,
            postfilter_model_type=args.postfilter_model_type,
            postfilter_model_path=args.postfilter_model_path,  # Will be expanded in method
            postfilter_model_types=args.postfilter_models,
            aggregation_method=args.aggregation_method,
            slide_model_type=args.slide_model_type,
            slide_model_path=model_paths.get(args.slide_model_type) if model_paths and args.slide_model_type else None,
            segment_threshold=args.segment_threshold,
            patch_size=args.patch_size,
            mpp=args.mpp,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            slide_batch_size=args.slide_batch_size,
            use_gpu=args.use_gpu,
            request_cpus=args.request_cpus,
            request_memory=args.request_memory,
            request_gpus=args.request_gpus if args.use_gpu else 0,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            aws_region=args.aws_region,
            hf_token=args.hf_token,
            max_retries=args.max_retries,
            model_paths=model_paths,  # Pass all model paths
            submit=args.submit,
        )


if __name__ == "__main__":
    main()
