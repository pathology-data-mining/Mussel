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

# Import model pre-download utility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
try:
    from model_predownload import pre_download_models
except ImportError:
    print("WARNING: Could not import model_predownload module. Pre-download features will be unavailable.")
    pre_download_models = None

try:
    from config_loader import load_config, load_config_defaults
except ImportError:
    print("WARNING: Could not import config_loader module. YAML config support will be unavailable.")
    load_config = None
    load_config_defaults = None


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
        partition: str = "batch",
        cpus_per_task: int = 4,
        mem: str = "16G",
        time: str = "02:00:00",
        gres: Optional[str] = None,
        qos: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = "us-east-1",
        aws_endpoint_url: Optional[str] = None,
        hf_token: Optional[str] = None,
        output_dir: Optional[str] = None,
        slide_batch_size: int = 8,
        **kwargs  # Accept and ignore extra parameters from config merging
    ) -> str:
        """Generate SLURM batch script content.
        
        Supports both single-slide and multi-slide batch processing.
        For batch processing, provide slide_paths and slide_ids instead of slide_path.
        """
        
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
            "NUM_WORKERS": str(num_workers),
            "BATCH_SIZE": str(batch_size),
            "USE_GPU": "true" if use_gpu else "false",
            "AWS_DEFAULT_REGION": aws_region,
        })
        
        # SegConfig group or individual parameters
        if seg_config_group:
            env_vars["SEG_CONFIG_GROUP"] = seg_config_group
        
        # Individual SegConfig parameters (only set if provided)
        if segment_threshold is not None:
            env_vars["SEGMENT_THRESHOLD"] = str(segment_threshold)
        if patch_size is not None:
            env_vars["PATCH_SIZE"] = str(patch_size)
        if step_size is not None:
            env_vars["STEP_SIZE"] = str(step_size)
        if mpp is not None:
            env_vars["MPP"] = str(mpp)
        if seg_level is not None:
            env_vars["SEG_LEVEL"] = str(seg_level)
        if segment_max_value is not None:
            env_vars["SEGMENT_MAX_VALUE"] = str(segment_max_value)
        if median_blur_ksize is not None:
            env_vars["MEDIAN_BLUR_KSIZE"] = str(median_blur_ksize)
        if morphology_ex_kernel is not None:
            env_vars["MORPHOLOGY_EX_KERNEL"] = str(morphology_ex_kernel)
        if ref_patch_size is not None:
            env_vars["REF_PATCH_SIZE"] = str(ref_patch_size)
        if use_otsu is not None:
            env_vars["USE_OTSU"] = "true" if use_otsu else "false"
        if tissue_area_threshold is not None:
            env_vars["TISSUE_AREA_THRESHOLD"] = str(tissue_area_threshold)
        if hole_area_threshold is not None:
            env_vars["HOLE_AREA_THRESHOLD"] = str(hole_area_threshold)
        if max_num_holes is not None:
            env_vars["MAX_NUM_HOLES"] = str(max_num_holes)
        
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

    def submit_tasks_from_config(
        self,
        config_file: str,
        **kwargs
    ) -> List[Optional[str]]:
        """
        Submit tasks from a configuration file (JSON or YAML).
        
        Config format:
            defaults:
                prefilter_model_type: CTRANSPATH
                batch_size: 64
                ...
            tasks:
                - task_id: task_1
                  slide_path: /path/to/slide1.svs
                  output_h5_path: /path/to/output1.h5
                  output_pt_path: /path/to/output1.pt
                - task_id: task_2
                  ...
        
        Args:
            config_file: Path to configuration file
            **kwargs: Additional parameters passed to submit_task
            
        Returns:
            List of job IDs
        """
        print(f"Loading task configuration from '{config_file}'...")
        
        # Use config loader to support both JSON and YAML
        if load_config:
            config = load_config(config_file)
        else:
            # Fallback to JSON only
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        tasks = config.get("tasks", [])
        defaults = config.get("defaults", {})
        
        print(f"Submitting {len(tasks)} tasks from config file...")
        
        job_ids = []
        for task_config in tasks:
            # Merge with defaults and kwargs
            merged_config = {**defaults, **kwargs, **task_config}
            
            task_id = merged_config.get("task_id")
            if not task_id:
                print("ERROR: task_id is required for each task in config file")
                continue
            
            slide_path = merged_config.get("slide_path")
            output_h5_path = merged_config.get("output_h5_path")
            output_pt_path = merged_config.get("output_pt_path")
            
            if not slide_path or not output_h5_path or not output_pt_path:
                print(f"ERROR: slide_path, output_h5_path, and output_pt_path required for task {task_id}")
                continue
            
            # Filter out keys that were extracted or don't belong to submit_task
            excluded_keys = ['task_id', 'slide_path', 'output_h5_path', 'output_pt_path', 'submit']
            
            # Submit task (use job_name instead of task_id for SLURM)
            job_id = self.submit_task(
                job_name=task_id,
                slide_path=slide_path,
                output_h5_path=output_h5_path,
                output_pt_path=output_pt_path,
                **{k: v for k, v in merged_config.items() if k not in excluded_keys}
            )
            job_ids.append(job_id)
        
        print(f"\nSubmitted {len(job_ids)} tasks from config file")
        return job_ids

    def submit_tasks_from_csv(
        self,
        csv_file: str,
        output_dir: Optional[str] = None,
        output_s3_prefix: Optional[str] = None,
        use_array: bool = True,
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
            use_array: If True, use SLURM job array. If False, submit individual jobs.
            distributed_slide_batch_size: Number of slides to group per task for batch encoding (default: 1).
                When > 1 and using slide-level model aggregation, slides are grouped into batches
                to optimize slide encoder loading. Recommended: 8-16 for GIGAPATH_SLIDE/TITAN_SLIDE.
            **kwargs: Additional task parameters
        
        Returns:
            List of job IDs
        """
        print(f"Reading CSV manifest: {csv_file}")
        
        # Read all slides
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
            
            # Group slides into batches
            job_ids = []
            for batch_idx in range(0, len(slides), distributed_slide_batch_size):
                batch_slides = slides[batch_idx:batch_idx + distributed_slide_batch_size]
                
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
                    job_name=batch_id,
                    slide_paths=slide_paths,
                    slide_ids=slide_ids,
                    output_dir_for_batch=output_dir_for_batch,
                    output_dir=kwargs.get('output_dir', 'slurm_logs'),
                    slide_batch_size=kwargs.get('slide_batch_size', 8),
                    **kwargs
                )
                job_ids.append(job_id)
            
            print(f"\nSubmitted {len(job_ids)} batch tasks")
            return job_ids
        
        # Original behavior: job array or individual jobs
        if use_array and len(slides) > 1:
            # Use job array
            return self._submit_job_array(slides, output_dir, output_s3_prefix, **kwargs)
        else:
            # Submit individual jobs
            job_ids = []
            for slide in slides:
                slide_id = slide['slide_id']
                slide_path = slide['slide_path']
                
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
        if kwargs.get('prefilter_model_path'):
            static_env["PREFILTER_MODEL_PATH"] = kwargs['prefilter_model_path']
        if kwargs.get('postfilter_model_type'):
            static_env["POSTFILTER_MODEL_TYPE"] = kwargs['postfilter_model_type']
        if kwargs.get('postfilter_model_path'):
            static_env["POSTFILTER_MODEL_PATH"] = kwargs['postfilter_model_path']
        if kwargs.get('postfilter_model_types'):
            static_env["POSTFILTER_MODEL_TYPES"] = kwargs['postfilter_model_types']
        if kwargs.get('aggregation_method'):
            static_env["AGGREGATION_METHOD"] = kwargs['aggregation_method']
        if kwargs.get('slide_model_type'):
            static_env["SLIDE_MODEL_TYPE"] = kwargs['slide_model_type']
        if kwargs.get('slide_model_path'):
            static_env["SLIDE_MODEL_PATH"] = kwargs['slide_model_path']
        if kwargs.get('aws_access_key_id'):
            static_env["AWS_ACCESS_KEY_ID"] = kwargs['aws_access_key_id']
        if kwargs.get('aws_secret_access_key'):
            static_env["AWS_SECRET_ACCESS_KEY"] = kwargs['aws_secret_access_key']
        if kwargs.get('aws_endpoint_url'):
            static_env["AWS_ENDPOINT_URL"] = kwargs['aws_endpoint_url']
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
    input_group.add_argument("--csv-manifest", help="CSV manifest file with slide_id,slide_path. "
                        "Can be used with --config to load parameters from config.")
    
    # Allow --config as optional parameter when using --csv-manifest
    parser.add_argument("--config", dest="config_file_for_csv",
                        help="Configuration file with default parameters (when using --csv-manifest)")
    
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
    
    # SegConfig parameters
    parser.add_argument("--seg-config-group", choices=["default", "biopsy", "resection", "tcga"],
                        help="SegConfig group preset (default, biopsy, resection, tcga). Overrides individual seg_config parameters.")
    parser.add_argument("--segment-threshold", type=int, help="Tissue segmentation threshold (default: 20 for default, varies by group)")
    parser.add_argument("--patch-size", type=int, help="Patch size in pixels (default: 256)")
    parser.add_argument("--step-size", type=int, help="Step size for patch extraction (default: same as patch_size)")
    parser.add_argument("--mpp", type=float, help="Microns per pixel (default: 0.5)")
    parser.add_argument("--seg-level", type=int, help="Segmentation pyramid level (default: -1 for auto)")
    parser.add_argument("--segment-max-value", type=int, help="Maximum pixel value for segmentation (default: 255)")
    parser.add_argument("--median-blur-ksize", type=int, help="Median blur kernel size (default: 7, varies by group)")
    parser.add_argument("--morphology-ex-kernel", type=int, help="Morphological closing kernel size (default: 0, varies by group)")
    parser.add_argument("--ref-patch-size", type=int, help="Reference patch size for thresholding (default: 512)")
    parser.add_argument("--use-otsu", action="store_true", help="Use Otsu thresholding (default: False)")
    parser.add_argument("--tissue-area-threshold", type=int, help="Tissue area threshold (default: 100, varies by group)")
    parser.add_argument("--hole-area-threshold", type=int, help="Hole area threshold (default: 16, varies by group)")
    parser.add_argument("--max-num-holes", type=int, help="Maximum number of holes (default: 8, varies by group)")
    
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
    
    # Submission
    parser.add_argument("--submit", action="store_true", help="Actually submit to SLURM")
    
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
    
    # Validate CTRANSPATH configuration
    # CTRANSPATH requires a model_path to be provided via configuration
    if args.csv_manifest:
        # Determine the prefilter model type from config or default
        prefilter_model = args.prefilter_model_type or 'CTRANSPATH'  # Default
        if args.config_file_for_csv and load_config_defaults:
            try:
                config_defaults = load_config_defaults(args.config_file_for_csv, backend='slurm')
                prefilter_model = config_defaults.get('prefilter_model_type', prefilter_model)
            except (FileNotFoundError, ValueError, KeyError, IOError):
                # If config loading fails, use default - validation warning will still show
                pass
        
        # Check if CTRANSPATH is being used without a model_path
        if prefilter_model.upper() == 'CTRANSPATH' and not model_paths.get('CTRANSPATH'):
            print("\n⚠️  WARNING: CTRANSPATH model requires a model_path to be provided via configuration")
            print("   CTRANSPATH does not have a default HuggingFace path and cannot be automatically downloaded.")
            print("   Please provide the model path using one of the following methods:")
            print("     1. Command line: --prefilter-model-path /path/to/ctranspath.pth")
            print("     2. Configuration file: prefilter_model_path: /path/to/ctranspath.pth")
            print("   Tasks will fail if CTRANSPATH model path is not provided.\n")
    
    # Validate single task arguments
    if args.job_name:
        if not args.slide_path or not args.output_h5_path or not args.output_pt_path:
            parser.error("--job-name requires --slide-path, --output-h5-path, and --output-pt-path")
    
    # Create submitter
    submitter = SlurmJobSubmitter()
    
    # Submit tasks
    if args.job_name:
        # Determine model paths for single task
        task_model_paths = {}
        if model_paths:
            task_model_paths = {
                'prefilter': model_paths.get('CTRANSPATH', args.prefilter_model_path),
                'postfilter': model_paths.get(args.postfilter_model_type, args.postfilter_model_path) if args.postfilter_model_type else None,
                'slide': model_paths.get(args.slide_model_type, args.slide_model_path) if args.slide_model_type else None,
            }
        
        submitter.submit_task(
            job_name=args.job_name,
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
            seg_config_group=args.seg_config_group,
            segment_threshold=args.segment_threshold,
            patch_size=args.patch_size,
            step_size=args.step_size,
            mpp=args.mpp,
            seg_level=args.seg_level,
            segment_max_value=args.segment_max_value,
            median_blur_ksize=args.median_blur_ksize,
            morphology_ex_kernel=args.morphology_ex_kernel,
            ref_patch_size=args.ref_patch_size,
            use_otsu=args.use_otsu,
            tissue_area_threshold=args.tissue_area_threshold,
            hole_area_threshold=args.hole_area_threshold,
            max_num_holes=args.max_num_holes,
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
        # CSV manifest (with optional config file for parameters)
        
        # Prepare kwargs dict starting with command-line args
        csv_kwargs = {
            'classifier_pkl': args.classifier_pkl,
            'classifier_threshold': args.classifier_threshold,
            'prefilter_model_type': args.prefilter_model_type,
            'postfilter_model_type': args.postfilter_model_type,
            'postfilter_model_types': args.postfilter_models,
            'aggregation_method': args.aggregation_method,
            'slide_model_type': args.slide_model_type,
            'seg_config_group': args.seg_config_group,
            'segment_threshold': args.segment_threshold,
            'patch_size': args.patch_size,
            'step_size': args.step_size,
            'mpp': args.mpp,
            'seg_level': args.seg_level,
            'segment_max_value': args.segment_max_value,
            'median_blur_ksize': args.median_blur_ksize,
            'morphology_ex_kernel': args.morphology_ex_kernel,
            'ref_patch_size': args.ref_patch_size,
            'use_otsu': args.use_otsu,
            'tissue_area_threshold': args.tissue_area_threshold,
            'hole_area_threshold': args.hole_area_threshold,
            'max_num_holes': args.max_num_holes,
            'num_workers': args.num_workers,
            'batch_size': args.batch_size,
            'slide_batch_size': args.slide_batch_size,
            'use_gpu': args.use_gpu,
            'partition': args.partition,
            'cpus_per_task': args.cpus_per_task,
            'mem': args.mem,
            'time': args.time,
            'gres': args.gres,
            'qos': args.qos,
            'aws_access_key_id': args.aws_access_key_id,
            'aws_secret_access_key': args.aws_secret_access_key,
            'aws_region': args.aws_region,
            'hf_token': args.hf_token,
            'submit': args.submit,
        }
        
        # If config file is provided for CSV, load defaults and merge
        if args.config_file_for_csv:
            if load_config_defaults:
                try:
                    print(f"Loading default parameters from config file: {args.config_file_for_csv}")
                    config_defaults = load_config_defaults(args.config_file_for_csv, backend='slurm')
                    # Config file defaults, then override with command-line args
                    merged_kwargs = {**config_defaults, **csv_kwargs}
                    csv_kwargs = merged_kwargs
                    print(f"Loaded {len(config_defaults)} default parameters from config file")
                except Exception as e:
                    print(f"WARNING: Failed to load config file: {e}")
                    print("Continuing with command-line parameters only")
            else:
                print("WARNING: config_loader not available, ignoring --config")
        
        # Add model paths from pre-download or user-provided (only if they have values)
        # These override config file values to ensure pre-downloaded models are used
        if model_paths and model_paths.get('CTRANSPATH'):
            csv_kwargs['prefilter_model_path'] = model_paths['CTRANSPATH']
        # Command-line args also override config values if explicitly provided
        if args.postfilter_model_path:
            csv_kwargs['postfilter_model_path'] = args.postfilter_model_path
        if model_paths and args.slide_model_type and model_paths.get(args.slide_model_type):
            csv_kwargs['slide_model_path'] = model_paths[args.slide_model_type]
        
        submitter.submit_tasks_from_csv(
            csv_file=args.csv_manifest,
            output_dir=args.output_dir,
            output_s3_prefix=args.output_s3_prefix,
            use_array=not args.no_array,
            distributed_slide_batch_size=args.distributed_slide_batch_size,
            **csv_kwargs,
        )


if __name__ == "__main__":
    main()
