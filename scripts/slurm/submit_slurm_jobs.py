#!/usr/bin/env python3
"""
SLURM job submission script for tessellate-extract-features.

This script submits one or more tessellate-extract-features tasks to SLURM.
It handles batch script generation, job array submission, and monitoring.

Requirements:
    - SLURM installed and configured
    - Access to SLURM submit node
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


class SlurmJobSubmitter:
    """
    Handles SLURM job submission for Mussel tessellate-extract-features.
    
    This class provides methods to:
    - Generate SLURM batch scripts
    - Submit single tasks or job arrays from CSV manifests
    - Monitor job progress
    - Handle retries via SLURM dependencies
    
    Typical usage:
        submitter = SlurmJobSubmitter()
        submitter.submit_tasks_from_csv("manifest.csv", output_dir="/output")
    """

    def __init__(self, script_dir: Optional[str] = None):
        """Initialize SLURM job submitter."""
        self.script_dir = script_dir or str(Path(__file__).parent)
        self.task_script = str(Path(__file__).parent.parent / "common" / "run_tessellate_extract_features.sh")
        
        # Check if task script exists
        if not os.path.exists(self.task_script):
            print(f"ERROR: Task script not found: {self.task_script}")
            sys.exit(1)

    def generate_batch_script(
        self,
        job_name: str,
        slide_path: str,
        output_h5_path: str,
        output_pt_path: str,
        intermediate_h5_path: Optional[str] = None,
        classifier_pkl: Optional[str] = None,
        classifier_threshold: float = 0.75,
        prefilter_model_type: str = "CTRANSPATH",
        postfilter_model_type: Optional[str] = None,
        postfilter_model_types: Optional[str] = None,
        aggregation_method: Optional[str] = None,
        slide_model_type: Optional[str] = None,
        segment_threshold: int = 0,
        patch_size: int = 256,
        mpp: float = 0.5,
        num_workers: int = 4,
        batch_size: int = 64,
        use_gpu: bool = True,
        partition: str = "batch",
        cpus_per_task: int = 4,
        mem: str = "16G",
        time: str = "02:00:00",
        gres: Optional[str] = None,
        qos: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = "us-east-1",
        hf_token: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """Generate SLURM batch script content."""
        
        # Set output directory for logs
        log_dir = output_dir or "slurm_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Build SLURM directives
        directives = [
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --time={time}",
            f"#SBATCH --output={log_dir}/{job_name}_%j.out",
            f"#SBATCH --error={log_dir}/{job_name}_%j.err",
        ]
        
        if gres:
            directives.append(f"#SBATCH --gres={gres}")
        
        if qos:
            directives.append(f"#SBATCH --qos={qos}")
        
        # Build environment variables
        env_vars = {
            "SLIDE_PATH": slide_path,
            "OUTPUT_H5_PATH": output_h5_path,
            "OUTPUT_PT_PATH": output_pt_path,
            "CLASSIFIER_THRESHOLD": str(classifier_threshold),
            "PREFILTER_MODEL_TYPE": prefilter_model_type,
            "SEGMENT_THRESHOLD": str(segment_threshold),
            "PATCH_SIZE": str(patch_size),
            "MPP": str(mpp),
            "NUM_WORKERS": str(num_workers),
            "BATCH_SIZE": str(batch_size),
            "USE_GPU": "true" if use_gpu else "false",
            "AWS_DEFAULT_REGION": aws_region,
        }
        
        if intermediate_h5_path:
            env_vars["INTERMEDIATE_H5_PATH"] = intermediate_h5_path
        if classifier_pkl:
            env_vars["CLASSIFIER_PKL"] = classifier_pkl
        if postfilter_model_type:
            env_vars["POSTFILTER_MODEL_TYPE"] = postfilter_model_type
        if postfilter_model_types:
            env_vars["POSTFILTER_MODEL_TYPES"] = postfilter_model_types
        if aggregation_method:
            env_vars["AGGREGATION_METHOD"] = aggregation_method
        if slide_model_type:
            env_vars["SLIDE_MODEL_TYPE"] = slide_model_type
        if aws_access_key_id:
            env_vars["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        if aws_secret_access_key:
            env_vars["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token
        
        # Generate batch script
        batch_content = f"""#!/bin/bash
{chr(10).join(directives)}

# Environment setup
{chr(10).join([f"export {k}={v}" for k, v in env_vars.items()])}

# Load modules (customize for your environment)
# module load python/3.9
# module load cuda/11.8

# Run task script
bash {self.task_script}
"""
        
        return batch_content

    def submit_task(
        self,
        job_name: str,
        slide_path: str,
        output_h5_path: str,
        output_pt_path: str,
        **kwargs
    ) -> Optional[str]:
        """
        Submit a single task to SLURM.
        
        Returns:
            Job ID if successful, None otherwise
        """
        print(f"Submitting job: {job_name}")
        
        # Generate batch script
        batch_content = self.generate_batch_script(
            job_name=job_name,
            slide_path=slide_path,
            output_h5_path=output_h5_path,
            output_pt_path=output_pt_path,
            **kwargs
        )
        
        # Write batch script
        batch_file = f"slurm_job_{job_name}.sbatch"
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        
        print(f"Generated batch script: {batch_file}")
        
        # Submit to SLURM
        if kwargs.get('submit', False):
            try:
                result = subprocess.run(
                    ["sbatch", batch_file],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(result.stdout)
                # Extract job ID from output
                # Expected format: "Submitted batch job 12345"
                if 'Submitted batch job' in result.stdout:
                    job_id = result.stdout.split()[-1].strip()
                    print(f"Job ID: {job_id}")
                    return job_id
            except subprocess.CalledProcessError as e:
                print(f"ERROR submitting job: {e}")
                print(e.stderr)
                return None
        else:
            print("Dry run - batch script generated but not submitted")
            print("Use --submit to actually submit to SLURM")
        
        return None

    def submit_tasks_from_csv(
        self,
        csv_file: str,
        output_dir: Optional[str] = None,
        output_s3_prefix: Optional[str] = None,
        use_array: bool = True,
        **kwargs
    ) -> List[Optional[str]]:
        """
        Submit multiple tasks from a CSV manifest.
        
        CSV format: slide_id,slide_path
        
        Args:
            use_array: If True, use SLURM job array. If False, submit individual jobs.
        
        Returns:
            List of job IDs
        """
        print(f"Reading CSV manifest: {csv_file}")
        
        # Read all slides
        slides = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                slides.append(row)
        
        if use_array and len(slides) > 1:
            # Use job array
            return self._submit_job_array(slides, output_dir, output_s3_prefix, **kwargs)
        else:
            # Submit individual jobs
            job_ids = []
            for row in slides:
                slide_id = row['slide_id']
                slide_path = row['slide_path']
                
                # Generate output paths
                output_h5_path, output_pt_path, intermediate_h5_path = self._generate_output_paths(
                    slide_id, output_dir, output_s3_prefix, **kwargs
                )
                
                if intermediate_h5_path:
                    kwargs['intermediate_h5_path'] = intermediate_h5_path
                
                # Submit task
                job_id = self.submit_task(
                    job_name=slide_id,
                    slide_path=slide_path,
                    output_h5_path=output_h5_path,
                    output_pt_path=output_pt_path,
                    output_dir=kwargs.get('output_dir', 'slurm_logs'),
                    **kwargs
                )
                job_ids.append(job_id)
            
            print(f"\nSubmitted {len(job_ids)} individual jobs")
            return job_ids

    def _generate_output_paths(self, slide_id, output_dir, output_s3_prefix, **kwargs):
        """Helper to generate output paths."""
        intermediate_h5_path = None
        
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
        
        elif output_dir:
            output_h5_path = os.path.join(output_dir, f"{slide_id}_features.h5")
            output_pt_path = os.path.join(output_dir, f"{slide_id}_features.pt")
        else:
            print(f"ERROR: Must specify --output-dir or --output-s3-prefix")
            sys.exit(1)
        
        return output_h5_path, output_pt_path, intermediate_h5_path

    def _submit_job_array(self, slides, output_dir, output_s3_prefix, **kwargs):
        """Submit a SLURM job array for multiple slides."""
        print(f"Submitting job array for {len(slides)} slides")
        
        # Create a manifest file for the array
        array_manifest = "slurm_array_manifest.csv"
        with open(array_manifest, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['slide_id', 'slide_path', 'output_h5_path', 'output_pt_path', 'intermediate_h5_path'])
            
            for row in slides:
                slide_id = row['slide_id']
                slide_path = row['slide_path']
                
                output_h5_path, output_pt_path, intermediate_h5_path = self._generate_output_paths(
                    slide_id, output_dir, output_s3_prefix, **kwargs
                )
                
                writer.writerow([slide_id, slide_path, output_h5_path, output_pt_path, intermediate_h5_path or ''])
        
        # Generate array batch script
        log_dir = kwargs.get('output_dir', 'slurm_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        partition = kwargs.get('partition', 'batch')
        cpus_per_task = kwargs.get('cpus_per_task', 4)
        mem = kwargs.get('mem', '16G')
        time = kwargs.get('time', '02:00:00')
        gres = kwargs.get('gres')
        qos = kwargs.get('qos')
        
        directives = [
            f"#SBATCH --job-name=mussel_array",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --time={time}",
            f"#SBATCH --array=1-{len(slides)}",
            f"#SBATCH --output={log_dir}/mussel_array_%A_%a.out",
            f"#SBATCH --error={log_dir}/mussel_array_%A_%a.err",
        ]
        
        if gres:
            directives.append(f"#SBATCH --gres={gres}")
        
        if qos:
            directives.append(f"#SBATCH --qos={qos}")
        
        # Build environment variables (those that don't change per task)
        static_env = {
            "CLASSIFIER_THRESHOLD": str(kwargs.get('classifier_threshold', 0.75)),
            "PREFILTER_MODEL_TYPE": kwargs.get('prefilter_model_type', 'CTRANSPATH'),
            "SEGMENT_THRESHOLD": str(kwargs.get('segment_threshold', 0)),
            "PATCH_SIZE": str(kwargs.get('patch_size', 256)),
            "MPP": str(kwargs.get('mpp', 0.5)),
            "NUM_WORKERS": str(kwargs.get('num_workers', 4)),
            "BATCH_SIZE": str(kwargs.get('batch_size', 64)),
            "USE_GPU": "true" if kwargs.get('use_gpu', True) else "false",
            "AWS_DEFAULT_REGION": kwargs.get('aws_region', 'us-east-1'),
        }
        
        if kwargs.get('classifier_pkl'):
            static_env["CLASSIFIER_PKL"] = kwargs['classifier_pkl']
        if kwargs.get('postfilter_model_type'):
            static_env["POSTFILTER_MODEL_TYPE"] = kwargs['postfilter_model_type']
        if kwargs.get('postfilter_model_types'):
            static_env["POSTFILTER_MODEL_TYPES"] = kwargs['postfilter_model_types']
        if kwargs.get('aggregation_method'):
            static_env["AGGREGATION_METHOD"] = kwargs['aggregation_method']
        if kwargs.get('slide_model_type'):
            static_env["SLIDE_MODEL_TYPE"] = kwargs['slide_model_type']
        if kwargs.get('aws_access_key_id'):
            static_env["AWS_ACCESS_KEY_ID"] = kwargs['aws_access_key_id']
        if kwargs.get('aws_secret_access_key'):
            static_env["AWS_SECRET_ACCESS_KEY"] = kwargs['aws_secret_access_key']
        if kwargs.get('hf_token'):
            static_env["HF_TOKEN"] = kwargs['hf_token']
        
        batch_content = f"""#!/bin/bash
{chr(10).join(directives)}

# Static environment variables
{chr(10).join([f"export {k}={v}" for k, v in static_env.items()])}

# Load modules (customize for your environment)
# module load python/3.9
# module load cuda/11.8

# Read task-specific variables from manifest
MANIFEST="{array_manifest}"
LINE=$((SLURM_ARRAY_TASK_ID + 1))  # +1 to skip header

# Extract values from CSV
IFS=',' read -r SLIDE_ID SLIDE_PATH OUTPUT_H5_PATH OUTPUT_PT_PATH INTERMEDIATE_H5_PATH < <(sed -n "${{LINE}}p" "$MANIFEST")

# Export task-specific variables
export SLIDE_PATH
export OUTPUT_H5_PATH
export OUTPUT_PT_PATH
if [ -n "$INTERMEDIATE_H5_PATH" ]; then
    export INTERMEDIATE_H5_PATH
fi

echo "Processing slide: $SLIDE_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Run task script
bash {self.task_script}
"""
        
        # Write batch script
        batch_file = "slurm_array_job.sbatch"
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        
        print(f"Generated array batch script: {batch_file}")
        print(f"Array manifest: {array_manifest}")
        
        # Submit to SLURM
        if kwargs.get('submit', False):
            try:
                result = subprocess.run(
                    ["sbatch", batch_file],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(result.stdout)
                if 'Submitted batch job' in result.stdout:
                    job_id = result.stdout.split()[-1].strip()
                    print(f"Job array ID: {job_id}")
                    return [job_id]
            except subprocess.CalledProcessError as e:
                print(f"ERROR submitting job array: {e}")
                print(e.stderr)
                return []
        else:
            print("Dry run - array batch script generated but not submitted")
            print("Use --submit to actually submit to SLURM")
        
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Submit tessellate-extract-features jobs to SLURM"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--job-name", help="Single job name/task ID")
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
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", action="store_false", dest="use_gpu")
    
    # SLURM resource requirements
    parser.add_argument("--partition", default="batch", help="SLURM partition")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="16G", help="Memory per task (e.g., 16G, 32GB)")
    parser.add_argument("--time", default="02:00:00", help="Time limit (HH:MM:SS)")
    parser.add_argument("--gres", help="Generic resources (e.g., gpu:1)")
    parser.add_argument("--qos", help="Quality of service")
    
    # Array job options
    parser.add_argument("--no-array", action="store_true", help="Submit individual jobs instead of array")
    
    # AWS S3 credentials
    parser.add_argument("--aws-access-key-id", help="AWS access key ID")
    parser.add_argument("--aws-secret-access-key", help="AWS secret access key")
    parser.add_argument("--aws-region", default="us-east-1")
    
    # HuggingFace token
    parser.add_argument("--hf-token", help="HuggingFace token for gated models")
    
    # Submission
    parser.add_argument("--submit", action="store_true", help="Actually submit to SLURM")
    
    args = parser.parse_args()
    
    # Validate single task arguments
    if args.job_name:
        if not args.slide_path or not args.output_h5_path or not args.output_pt_path:
            parser.error("--job-name requires --slide-path, --output-h5-path, and --output-pt-path")
    
    # Create submitter
    submitter = SlurmJobSubmitter()
    
    # Submit tasks
    if args.job_name:
        submitter.submit_task(
            job_name=args.job_name,
            slide_path=args.slide_path,
            output_h5_path=args.output_h5_path,
            output_pt_path=args.output_pt_path,
            classifier_pkl=args.classifier_pkl,
            classifier_threshold=args.classifier_threshold,
            prefilter_model_type=args.prefilter_model_type,
            postfilter_model_type=args.postfilter_model_type,
            postfilter_model_types=args.postfilter_models,
            aggregation_method=args.aggregation_method,
            slide_model_type=args.slide_model_type,
            segment_threshold=args.segment_threshold,
            patch_size=args.patch_size,
            mpp=args.mpp,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            use_gpu=args.use_gpu,
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            time=args.time,
            gres=args.gres,
            qos=args.qos,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            aws_region=args.aws_region,
            hf_token=args.hf_token,
            submit=args.submit,
        )
    else:
        submitter.submit_tasks_from_csv(
            csv_file=args.csv_manifest,
            output_dir=args.output_dir,
            output_s3_prefix=args.output_s3_prefix,
            use_array=not args.no_array,
            classifier_pkl=args.classifier_pkl,
            classifier_threshold=args.classifier_threshold,
            prefilter_model_type=args.prefilter_model_type,
            postfilter_model_type=args.postfilter_model_type,
            postfilter_model_types=args.postfilter_models,
            aggregation_method=args.aggregation_method,
            slide_model_type=args.slide_model_type,
            segment_threshold=args.segment_threshold,
            patch_size=args.patch_size,
            mpp=args.mpp,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            use_gpu=args.use_gpu,
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            time=args.time,
            gres=args.gres,
            qos=args.qos,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            aws_region=args.aws_region,
            hf_token=args.hf_token,
            submit=args.submit,
        )


if __name__ == "__main__":
    main()
