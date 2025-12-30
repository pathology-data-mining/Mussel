#!/usr/bin/env python3
"""
SLURM job submission script for tessellate-extract-features.

This script submits one or more tessellate-extract-features tasks to SLURM.
It handles batch script generation, job array submission, and monitoring.

Requirements:
    - SLURM installed and configured
    - Access to SLURM submit node

Configuration:
    This script uses configargparse for flexible configuration management.
    You can provide configuration via:
    1. YAML config file: -c/--config config.yaml
    2. Command-line arguments (override config file)
    3. Default config files: slurm_config.yaml or slurm_config.yml

    Example usage:
        # Using config file
        python submit_slurm_jobs.py -c my_config.yaml --csv-manifest slides.csv --submit

        # Override config with command-line args
        python submit_slurm_jobs.py -c my_config.yaml --csv-manifest slides.csv --batch-size 512 --submit

    See scripts/slurm/slurm_config_example.yaml for a config file template.
"""

import configargparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Import model pre-download utility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
try:
    from model_predownload import pre_download_models
except ImportError:
    print(
        "WARNING: Could not import model_predownload module. Pre-download features will be unavailable."
    )
    pre_download_models = None

try:
    from config_loader import load_config, load_config_defaults
except ImportError:
    print(
        "WARNING: Could not import config_loader module. YAML config support will be unavailable."
    )
    load_config = None
    load_config_defaults = None


class FlatteningYAMLConfigFileParser(configargparse.YAMLConfigFileParser):
    """
    Custom YAML config file parser that flattens nested structures.

    This parser extends the default YAMLConfigFileParser to handle nested YAML
    structures like seg_config.group and converts them to flat keys like
    seg_config_group that configargparse can understand.
    """

    def parse(self, stream):
        """Parse YAML config file and flatten nested structures."""
        try:
            import yaml
        except ImportError:
            raise configargparse.ConfigFileParserException(
                "PyYAML library is required to parse YAML config files"
            )

        try:
            config = yaml.safe_load(stream)
        except Exception as e:
            raise configargparse.ConfigFileParserException(f"Couldn't parse YAML: {e}")

        if config is None:
            return {}

        # Flatten nested structures
        flattened = {}

        # Handle seg_config section
        if "seg_config" in config and isinstance(config["seg_config"], dict):
            seg_config = config["seg_config"]
            if "group" in seg_config:
                flattened["seg_config_group"] = str(seg_config["group"])
            # Add other seg_config parameters as top-level
            for key, value in seg_config.items():
                if key != "group":
                    flattened[key] = (
                        str(value) if not isinstance(value, (list, dict)) else value
                    )
            # Remove seg_config from main config
            config = {k: v for k, v in config.items() if k != "seg_config"}

        # Handle aws section
        if "aws" in config and isinstance(config["aws"], dict):
            aws_config = config["aws"]
            if "region" in aws_config:
                flattened["aws_region"] = str(aws_config["region"])
            if "endpoint_url" in aws_config:
                flattened["aws_endpoint_url"] = str(aws_config["endpoint_url"])
            if "access_key_id" in aws_config:
                flattened["aws_access_key_id"] = str(aws_config["access_key_id"])
            if "secret_access_key" in aws_config:
                flattened["aws_secret_access_key"] = str(
                    aws_config["secret_access_key"]
                )
            # Remove aws from main config
            config = {k: v for k, v in config.items() if k != "aws"}

        # Handle resources section
        if "resources" in config and isinstance(config["resources"], dict):
            resources = config["resources"]
            if "cpus" in resources:
                flattened["cpus_per_task"] = str(resources["cpus"])
            if "memory" in resources:
                flattened["mem"] = str(resources["memory"])
            if "gpus" in resources:
                flattened["gres"] = f"gpu:{resources['gpus']}"
            # Remove resources from main config
            config = {k: v for k, v in config.items() if k != "resources"}

        # Handle slurm section
        if "slurm" in config and isinstance(config["slurm"], dict):
            # Convert all values to strings
            for key, value in config["slurm"].items():
                flattened[key] = (
                    str(value) if not isinstance(value, (list, dict)) else value
                )
            # Remove slurm from main config
            config = {k: v for k, v in config.items() if k != "slurm"}

        # Normalize list parameters to comma-separated strings
        list_params = ["model_types", "slide_model_types", "prefilter_model_types"]
        for param in list_params:
            if param in config and isinstance(config[param], list):
                config[param] = ",".join(str(item) for item in config[param])

        # Handle model_types/slide_model_types conversion
        if "model_types" in config:
            flattened["models"] = str(config["model_types"])
            config = {k: v for k, v in config.items() if k != "model_types"}
        if "slide_model_types" in config:
            flattened["slide_models"] = str(config["slide_model_types"])
            config = {k: v for k, v in config.items() if k != "slide_model_types"}

        # Remove backend-specific sections that aren't relevant for SLURM
        # Also remove extra parameters that aren't used by SLURM submission
        config = {
            k: v
            for k, v in config.items()
            if k
            not in [
                "azure",
                "condor",
                "azure_batch",
                "tasks",
                "model_batch_sizes",
                "slides_per_task",
                "max_retry_count",
            ]
        }

        # Convert remaining values to strings (required by configargparse)
        for key, value in config.items():
            if not isinstance(value, (list, dict)):
                flattened[key] = str(value)
            else:
                # Keep lists and dicts as-is for special handling
                flattened[key] = value

        # Convert underscores to dashes in keys (configargparse expects dashes)
        normalized = {}
        for key, value in flattened.items():
            normalized_key = key.replace("_", "-")
            normalized[normalized_key] = value

        return normalized


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
        self.task_script = str(
            Path(__file__).parent.parent
            / "common"
            / "run_tessellate_extract_features.sh"
        )

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
        output_dir_for_batch: Optional[str] = None,
        intermediate_h5_path: Optional[str] = None,
        classifier_pkl: Optional[str] = None,
        classifier_threshold: float = 0.75,
        prefilter_model_type: Optional[str] = None,
        prefilter_model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        model_path: Optional[str] = None,
        model_types: Optional[str] = None,
        aggregation_method: Optional[str] = None,
        slide_model_type: Optional[str] = None,
        slide_model_path: Optional[str] = None,
        slide_model_types: Optional[str] = None,
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
        batch_size: int = 256,  # Conservative default to avoid OOM
        use_gpu: bool = True,
        partition: Optional[str] = None,
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
        use_docker: bool = False,
        docker_image: str = "ghcr.io/biomedia-mira/mussel:latest",
        docker_runtime: str = "nvidia",
        container_runtime: str = "singularity",  # 'docker', 'singularity', or 'apptainer'
        mount_code_dir: Optional[str] = None,  # Local code directory to mount
        **kwargs,  # Accept and ignore extra parameters from config merging
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
        ]

        if partition:
            directives.append(f"#SBATCH --partition={partition}")

        directives.extend(
            [
                f"#SBATCH --cpus-per-task={cpus_per_task}",
                f"#SBATCH --mem={mem}",
                f"#SBATCH --time={time}",
                f"#SBATCH --output={log_dir}/{job_name}_%j.out",
                f"#SBATCH --error={log_dir}/{job_name}_%j.err",
            ]
        )

        if gres:
            directives.append(f"#SBATCH --gres={gres}")

        if qos:
            directives.append(f"#SBATCH --qos={qos}")

        # Build environment variables
        env_vars = {}

        # Handle batch vs single slide processing
        if slide_paths and len(slide_paths) >= 1:
            # Batch processing mode (including single-slide batches)
            # Use pipe delimiter to avoid conflicts with commas in S3 paths (e.g., paths with spaces)
            env_vars["SLIDE_PATHS"] = "|".join(slide_paths)
            if slide_ids:
                env_vars["SLIDE_IDS"] = ",".join(slide_ids)
            if output_dir_for_batch:
                # For container execution, OUTPUT_DIR should point to container path
                # For direct execution, use absolute host path
                if use_docker:
                    env_vars["OUTPUT_DIR"] = "/workspace/output"
                else:
                    env_vars["OUTPUT_DIR"] = os.path.abspath(output_dir_for_batch)
            env_vars["SLIDE_BATCH_SIZE"] = str(slide_batch_size)
        else:
            # Single slide mode (backward compatible - non-batch)
            env_vars["SLIDE_PATH"] = slide_path
            if output_dir:
                # For container execution, OUTPUT_DIR should point to container path
                # For direct execution, use absolute host path
                if use_docker:
                    env_vars["OUTPUT_DIR"] = "/workspace/output"
                else:
                    env_vars["OUTPUT_DIR"] = os.path.abspath(output_dir)

        # Common environment variables
        env_vars.update(
            {
                "CLASSIFIER_THRESHOLD": str(classifier_threshold),
                "NUM_WORKERS": str(num_workers),
                "BATCH_SIZE": str(batch_size),
                "USE_GPU": "true" if use_gpu else "false",
                "AWS_DEFAULT_REGION": aws_region,
            }
        )

        # Only add prefilter if specified
        if prefilter_model_type:
            env_vars["PREFILTER_MODEL_TYPE"] = prefilter_model_type

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
        if model_type:
            env_vars["MODEL_TYPE"] = model_type
        if model_path:
            env_vars["MODEL_PATH"] = model_path
        if model_types:
            env_vars["MODEL_TYPES"] = model_types
        if aggregation_method:
            env_vars["AGGREGATION_METHOD"] = aggregation_method
        if slide_model_type:
            env_vars["SLIDE_MODEL_TYPE"] = slide_model_type
        if slide_model_types:
            env_vars["SLIDE_MODEL_TYPES"] = slide_model_types
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

        # Add MODEL_DIR and HuggingFace cache dirs
        # For container execution, use container paths
        # For direct execution, use host paths
        model_dir_value = kwargs.get("model_dir", "./model_cache")
        if use_docker:
            # Container paths
            env_vars["MODEL_DIR"] = "/workspace/model_cache"
            env_vars["HF_HOME"] = "/workspace/model_cache/.cache/huggingface"
            env_vars["TRANSFORMERS_CACHE"] = "/workspace/model_cache/.cache/huggingface"
        else:
            # Host paths
            abs_model_dir = os.path.abspath(model_dir_value)
            env_vars["MODEL_DIR"] = abs_model_dir
            env_vars["HF_HOME"] = f"{abs_model_dir}/.cache/huggingface"
            env_vars["TRANSFORMERS_CACHE"] = f"{abs_model_dir}/.cache/huggingface"

        # If mounting code directory, set PYTHONPATH to use mounted code
        if mount_code_dir and use_docker:
            env_vars["PYTHONPATH"] = "/app:$PYTHONPATH"

        # Generate batch script
        batch_content = f"""#!/bin/bash
{chr(10).join(directives)}

# Environment setup
{chr(10).join([f'export {k}="{v}"' for k, v in env_vars.items()])}

# Set up tmp directory for Singularity/Apptainer and Python
export TMPDIR=$HOME/tmp
export SINGULARITY_TMPDIR=$HOME/tmp
export APPTAINER_TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

# Load modules (customize for your environment)
# module load python/3.9
# module load cuda/11.8

# OOM retry logic
# Store original memory request
ORIGINAL_MEM="{mem}"
RETRY_COUNT_FILE="$TMPDIR/.retry_count_${{SLURM_JOB_ID}}"
if [ -f "$RETRY_COUNT_FILE" ]; then
    RETRY_COUNT=$(cat "$RETRY_COUNT_FILE")
else
    RETRY_COUNT=0
fi

# Run task script
"""

        if use_docker:
            # Container execution mode
            if container_runtime in ("singularity", "apptainer"):
                batch_content += self._generate_singularity_command(
                    env_vars=env_vars,
                    docker_image=docker_image,
                    output_dir=output_dir_for_batch or output_dir,
                    model_dir=kwargs.get("model_dir", "./model_cache"),
                    slide_paths=slide_paths,
                    slide_path=slide_path,
                    runtime=container_runtime,
                    mount_code_dir=mount_code_dir,
                )
            else:
                # Docker execution mode
                batch_content += self._generate_docker_command(
                    env_vars=env_vars,
                    docker_image=docker_image,
                    docker_runtime=docker_runtime,
                    output_dir=output_dir_for_batch or output_dir,
                    model_dir=kwargs.get("model_dir", "./model_cache"),
                    slide_paths=slide_paths,
                    slide_path=slide_path,
                    mount_code_dir=mount_code_dir,
                )
        else:
            # Direct execution mode
            batch_content += f"bash {self.task_script}\n"

        # Add OOM retry logic at the end
        batch_content += """
# Capture exit code
EXIT_CODE=$?

# Check if job was killed due to OOM
if [ $EXIT_CODE -ne 0 ]; then
    # Check SLURM sacct for OOM signal
    sleep 2  # Wait for sacct to update
    OOM_CHECK=$(sacct -j $SLURM_JOB_ID --format=State -n | grep -i "OUT_OF_MEMORY\\|OOM")
    
    # Also check dmesg for OOM killer messages
    DMESG_OOM=$(dmesg | tail -100 | grep -i "killed process.*$SLURM_JOB_ID" || true)
    
    if [ -n "$OOM_CHECK" ] || [ -n "$DMESG_OOM" ] || [ $EXIT_CODE -eq 137 ]; then
        echo "================================================"
        echo "JOB KILLED DUE TO OUT-OF-MEMORY (OOM)"
        echo "================================================"
        
        # Limit retries to 3 attempts
        if [ $RETRY_COUNT -lt 3 ]; then
            NEW_RETRY_COUNT=$((RETRY_COUNT + 1))
            echo $NEW_RETRY_COUNT > "$RETRY_COUNT_FILE"
            
            # Calculate new memory requirement (multiply by 1.5x each retry)
            # Parse memory value and unit
            MEM_VALUE=$(echo "$ORIGINAL_MEM" | grep -oP '\\d+')
            MEM_UNIT=$(echo "$ORIGINAL_MEM" | grep -oP '[A-Za-z]+$')
            
            # Multiply by 1.5 for each retry
            MULTIPLIER=$(awk "BEGIN {print 1.5 ^ $NEW_RETRY_COUNT}")
            NEW_MEM_VALUE=$(awk "BEGIN {printf \\"%.0f\\", $MEM_VALUE * $MULTIPLIER}")
            NEW_MEM="${NEW_MEM_VALUE}${MEM_UNIT}"
            
            echo "Retry attempt: $NEW_RETRY_COUNT / 3"
            echo "Original memory: $ORIGINAL_MEM"
            echo "New memory request: $NEW_MEM"
            echo "Resubmitting job with increased memory..."
            
            # Resubmit the same job with increased memory
            # Use sbatch with --dependency to ensure it runs after this job ends
            NEW_JOB_ID=$(sbatch --mem=$NEW_MEM \\
                --dependency=afterany:$SLURM_JOB_ID \\
                --export=ALL \\
                $0 | awk '{print $NF}')
            
            echo "Resubmitted as job ID: $NEW_JOB_ID"
            exit 0  # Exit gracefully to allow retry
        else
            echo "Maximum retry attempts (3) reached. Job failed permanently."
            echo "Consider manually increasing memory or optimizing the workload."
            exit 1
        fi
    fi
fi

# Clean up retry count file on success
if [ $EXIT_CODE -eq 0 ] && [ -f "$RETRY_COUNT_FILE" ]; then
    rm -f "$RETRY_COUNT_FILE"
fi

exit $EXIT_CODE
"""

        return batch_content

    def _generate_docker_command(
        self,
        env_vars: dict,
        docker_image: str,
        docker_runtime: str,
        output_dir: Optional[str],
        model_dir: str,
        slide_paths: Optional[List[str]] = None,
        slide_path: Optional[str] = None,
        mount_code_dir: Optional[str] = None,
    ) -> str:
        """Generate Docker command with proper volume mounts and environment variables."""

        # Convert to absolute paths
        abs_output_dir = os.path.abspath(output_dir) if output_dir and not output_dir.startswith("s3://") else output_dir
        abs_model_dir = os.path.abspath(model_dir)
        
        # Build docker run command
        docker_cmd = "docker run --rm \\\n"
        docker_cmd += f"  --runtime={docker_runtime} \\\n"
        docker_cmd += "  --shm-size=8g \\\n"  # Increase shared memory for PyTorch dataloaders

        # Add environment variables
        for key, value in env_vars.items():
            # Escape special characters in environment variable values
            escaped_value = str(value).replace('"', '\\"').replace("$", "\\$")
            docker_cmd += f'  -e {key}="{escaped_value}" \\\n'

        # Mount $HOME/tmp for temporary files
        docker_cmd += "  -v $HOME/tmp:$HOME/tmp \\\n"
        docker_cmd += "  -e TMPDIR=$HOME/tmp \\\n"

        # Add model directory mount (always needed)
        docker_cmd += f"  -v {abs_model_dir}:/workspace/model_cache \\\n"

        # Add output directory mount if local
        if abs_output_dir and not abs_output_dir.startswith("s3://"):
            docker_cmd += f"  -v {abs_output_dir}:/workspace/output \\\n"

        # Add code directory mount if specified (for development)
        if mount_code_dir:
            abs_code_dir = os.path.abspath(mount_code_dir)
            docker_cmd += f"  -v {abs_code_dir}:/app \\\n"

        # Add slide path mounts
        if slide_paths:
            # Multiple slides - mount parent directories
            for sp in slide_paths:
                if not sp.startswith("s3://"):
                    docker_cmd += (
                        f"  -v $(dirname $(realpath {sp})):/workspace/slides \\\n"
                    )
        elif slide_path and not slide_path.startswith("s3://"):
            # Single slide - mount parent directory
            docker_cmd += (
                f"  -v $(dirname $(realpath {slide_path})):/workspace/slides \\\n"
            )

        # Add the Docker image and command
        docker_cmd += f"  {docker_image} \\\n"
        docker_cmd += f"  bash /app/scripts/common/run_tessellate_extract_features.sh\n"

        return docker_cmd

    def _generate_singularity_command(
        self,
        env_vars: dict,
        docker_image: str,
        output_dir: Optional[str],
        model_dir: str,
        slide_paths: Optional[List[str]] = None,
        slide_path: Optional[str] = None,
        runtime: str = "singularity",
        mount_code_dir: Optional[str] = None,
    ) -> str:
        """Generate Singularity/Apptainer command with proper bind mounts and environment variables."""

        # Check if docker_image is a local SIF file
        is_local_sif = docker_image.endswith(".sif") or os.path.isfile(docker_image)

        # Convert Docker image to Singularity SIF format if not a local file
        # Format: docker://registry/image:tag
        if not is_local_sif and not docker_image.startswith("docker://"):
            docker_image = f"docker://{docker_image}"

        # Create directories that need to exist before binding
        # Convert to absolute paths to avoid issues with relative paths
        cmd = ""
        abs_output_dir = os.path.abspath(output_dir) if output_dir and not output_dir.startswith("s3://") else output_dir
        abs_model_dir = os.path.abspath(model_dir)
        
        if abs_output_dir and not abs_output_dir.startswith("s3://"):
            cmd += f"mkdir -p {abs_output_dir}\n"
        cmd += f"mkdir -p {abs_model_dir}\n\n"

        # Build singularity exec command
        cmd += f"{runtime} exec --nv \\\n"  # --nv for NVIDIA GPU support
        cmd += "  --no-home \\\n"  # Avoid home directory conflicts
        cmd += "  --containall \\\n"  # More isolation to avoid conflicts
        
        # Increase shared memory to avoid OOM with PyTorch dataloaders
        # Most SLURM nodes have at least 16GB of RAM, so 8GB shm should be safe
        cmd += "  --env APPTAINER_SHM_SIZE=8G \\\n"
        cmd += "  --env SINGULARITY_SHM_SIZE=8G \\\n"

        # Add environment variables
        for key, value in env_vars.items():
            # Escape special characters in environment variable values
            escaped_value = str(value).replace('"', '\\"').replace("$", "\\$")
            cmd += f'  --env {key}="{escaped_value}" \\\n'

        # Bind mount $HOME/tmp for temporary files
        cmd += "  --bind $HOME/tmp:$HOME/tmp \\\n"
        cmd += "  --env TMPDIR=$HOME/tmp \\\n"

        # Add model directory bind mount (always needed)
        cmd += f"  --bind {abs_model_dir}:/workspace/model_cache \\\n"

        # Add output directory bind mount if local
        if abs_output_dir and not abs_output_dir.startswith("s3://"):
            cmd += f"  --bind {abs_output_dir}:/workspace/output \\\n"

        # Add code directory bind mount if specified (for development)
        if mount_code_dir:
            cmd += f"  --bind $(realpath {mount_code_dir}):/app \\\n"

        # Collect unique parent directories for slides
        bind_dirs = set()
        if slide_paths:
            for sp in slide_paths:
                if not sp.startswith("s3://"):
                    # Get parent directory
                    parent = os.path.dirname(sp)
                    bind_dirs.add(parent)
        elif slide_path and not slide_path.startswith("s3://"):
            parent = os.path.dirname(slide_path)
            bind_dirs.add(parent)

        # Add slide directory bind mounts
        for bind_dir in bind_dirs:
            cmd += f"  --bind {bind_dir}:{bind_dir} \\\n"

        # Add the container image and command
        cmd += f"  {docker_image} \\\n"
        cmd += f"  bash /app/scripts/common/run_tessellate_extract_features.sh\n"

        return cmd

    def submit_task(
        self, job_name: str, slide_path: str = None, **kwargs
    ) -> Optional[str]:
        """
        Submit a single task to SLURM.

        Supports both single-slide and multi-slide batch processing:
        - Single slide: Provide slide_path and output_dir
        - Batch mode: Provide slide_paths, slide_ids, output_dir_for_batch in kwargs

        Returns:
            Job ID if successful, None otherwise
        """
        print(f"Submitting job: {job_name}")

        # Generate batch script
        batch_content = self.generate_batch_script(
            job_name=job_name, slide_path=slide_path, **kwargs
        )

        # Write batch script
        batch_file = f"slurm_job_{job_name}.sbatch"
        with open(batch_file, "w") as f:
            f.write(batch_content)

        print(f"Generated batch script: {batch_file}")

        # Submit to SLURM
        if kwargs.get("submit", False):
            try:
                result = subprocess.run(
                    ["sbatch", batch_file], capture_output=True, text=True, check=True
                )
                print(result.stdout)
                # Extract job ID from output
                # Expected format: "Submitted batch job 12345"
                if "Submitted batch job" in result.stdout:
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

        Batch encoding is beneficial when processing multiple slides because:
        1. Patch encoder model is loaded once instead of N times
        2. Better GPU utilization through batched processing
        3. If using slide-level aggregation, slide encoder is also loaded once

        Returns True for any multi-slide processing scenario.
        """
        # Batch encoding is beneficial whenever processing multiple slides
        # The CLI will handle both patch extraction and slide aggregation efficiently
        return True

    def submit_tasks_from_config(
        self, config_file: str, **kwargs
    ) -> List[Optional[str]]:
        """
        Submit tasks from a configuration file (JSON or YAML).

        Config format:
            defaults:
                prefilter_model_type: CTRANSPATH
                batch_size: 64
                output_dir: /path/to/output
                ...
            tasks:
                - task_id: task_1
                  slide_path: /path/to/slide1.svs
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

            with open(config_file, "r") as f:
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
            output_dir = merged_config.get("output_dir")

            if not slide_path or not output_dir:
                print(f"ERROR: slide_path and output_dir required for task {task_id}")
                continue

            # Normalize empty string to None for intermediate_h5_path
            intermediate_h5_path = merged_config.get("intermediate_h5_path") or None

            # Filter out keys that were extracted or don't belong to submit_task
            excluded_keys = ["task_id", "slide_path", "submit"]

            # Submit task (use job_name instead of task_id for SLURM)
            job_id = self.submit_task(
                job_name=task_id,
                slide_path=slide_path,
                intermediate_h5_path=intermediate_h5_path,
                **{
                    k: v
                    for k, v in merged_config.items()
                    if k not in excluded_keys and k != "intermediate_h5_path"
                },
            )
            job_ids.append(job_id)

        print(f"\nSubmitted {len(job_ids)} tasks from config file")
        return job_ids

    def submit_tasks_from_csv(
        self,
        csv_file: str,
        output_dir: Optional[str] = None,
        distributed_slide_batch_size: Optional[int] = None,
        **kwargs,
    ) -> List[Optional[str]]:
        """
        Submit multiple tasks from a CSV manifest.

        CSV format: slide_id,slide_path

        Args:
            csv_file: Path to CSV manifest file
            output_dir: Output directory for results (can be local path or S3 path)
            distributed_slide_batch_size: Number of slides to group per task for batch encoding (default: auto).
                When None (auto), automatically enables batching with size=8 for multi-slide processing.
                Set to 1 to explicitly disable batching (process one slide per task).
                Recommended: 8-16 for GIGAPATH_SLIDE/TITAN_SLIDE.
            **kwargs: Additional task parameters

        Returns:
            List of job IDs
        """
        print(f"Reading CSV manifest: {csv_file}")

        # Read all slides
        slides = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support multiple column name conventions
                slide_id = (
                    row.get("slide_id") or row.get("sample_id") or row.get("image_id")
                )
                slide_path = (
                    row.get("slide_path") or row.get("svs_path") or row.get("path")
                )

                if not slide_id or not slide_path:
                    raise ValueError(f"Row missing required columns: {row}")

                slides.append({"slide_id": slide_id, "slide_path": slide_path})

        # Auto-adjust distributed_slide_batch_size if not explicitly set (None)
        # and slide-level model aggregation is enabled
        if distributed_slide_batch_size is None and self._should_use_batch_encoding(
            **kwargs
        ):
            distributed_slide_batch_size = 8  # Recommended default for batch encoding
            print(f"\n[Auto-Batching] Detected multi-slide processing")
            print(
                f"  Automatically enabling batch processing with batch_size={distributed_slide_batch_size}"
            )
            print(f"  (Use --distributed-slide-batch-size 1 to disable, or --distributed-slide-batch-size N to customize)")
        elif distributed_slide_batch_size is None:
            # No batching conditions met, default to 1 (no batching)
            distributed_slide_batch_size = 1

        # Determine if we should use batch encoding
        use_batch_encoding = (
            distributed_slide_batch_size > 1
            and self._should_use_batch_encoding(**kwargs)
        )

        if use_batch_encoding:
            print(f"\n[Batch Encoding Optimization] Enabled")
            print(f"  Grouping slides into batches of {distributed_slide_batch_size}")
            print(f"  Slide encoder: {kwargs.get('slide_model_type')}")
            print(
                f"  This reduces model loading overhead from {len(slides)}x to {(len(slides) + distributed_slide_batch_size - 1) // distributed_slide_batch_size}x"
            )

            # Group slides into batches
            job_ids = []
            for batch_idx in range(0, len(slides), distributed_slide_batch_size):
                batch_slides = slides[
                    batch_idx : batch_idx + distributed_slide_batch_size
                ]

                # Create batch task ID
                batch_id = f"batch_{batch_idx // distributed_slide_batch_size + 1}_of_{(len(slides) + distributed_slide_batch_size - 1) // distributed_slide_batch_size}"

                # Extract slide IDs and paths for this batch
                slide_ids = [s["slide_id"] for s in batch_slides]
                slide_paths = [s["slide_path"] for s in batch_slides]

                print(f"\nSubmitting batch task: {batch_id}")
                print(f"  Slides: {', '.join(slide_ids)}")

                # Determine output directory for batch
                if not output_dir:
                    print(f"ERROR: Must specify --output-dir")
                    sys.exit(1)

                # If output_dir starts with s3://, organize by model type
                if output_dir.startswith("s3://"):
                    model_type = kwargs.get("prefilter_model_type")
                    if not model_type and kwargs.get("model_types"):
                        models = kwargs["model_types"].split(",")
                        model_type = models[0]
                    if not model_type and kwargs.get("model_type"):
                        model_type = kwargs["model_type"]
                    output_dir_for_batch = f"{output_dir.rstrip('/')}/{model_type}"
                else:
                    output_dir_for_batch = output_dir

                # Submit batch task
                slide_batch_size = kwargs.get("slide_batch_size", 8)
                # Exclude parameters that are passed explicitly
                filtered_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["slide_batch_size", "output_dir"]
                }

                job_id = self.submit_task(
                    job_name=batch_id,
                    slide_paths=slide_paths,
                    slide_ids=slide_ids,
                    output_dir_for_batch=output_dir_for_batch,
                    output_dir=kwargs.get("output_dir", "slurm_logs"),
                    slide_batch_size=slide_batch_size,
                    **filtered_kwargs,
                )
                job_ids.append(job_id)

            print(f"\nSubmitted {len(job_ids)} batch tasks")
            return job_ids

        # Individual job submission (when distributed_slide_batch_size == 1 or batch encoding disabled)
        # Note: Job array submission is deprecated and removed
        job_ids = []
        for slide in slides:
            slide_id = slide["slide_id"]
            slide_path = slide["slide_path"]

            # Submit task
            job_id = self.submit_task(
                job_name=slide_id,
                slide_path=slide_path,
                output_dir=output_dir,
                **kwargs,
            )
            job_ids.append(job_id)

        print(f"\nSubmitted {len(job_ids)} individual jobs")
        return job_ids

    def _submit_job_array(self, slides, output_dir, **kwargs):
        """Job array submission removed - only individual job submission supported."""
        print("ERROR: Job array submission is no longer supported.")
        print("Please use individual job submission by removing --use-job-array flag.")
        sys.exit(1)


def main():
    parser = configargparse.ArgumentParser(
        description="Submit tessellate-extract-features jobs to SLURM",
        default_config_files=["slurm_config.yaml", "slurm_config.yml"],
        config_file_parser_class=FlatteningYAMLConfigFileParser,
    )

    # Config file option
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Config file path (YAML format)"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--job-name", help="Single job name/task ID")
    input_group.add_argument(
        "--csv-manifest", help="CSV manifest file with slide_id,slide_path"
    )

    # Slide parameters
    parser.add_argument(
        "--slide-path", help="Path to slide file (required for single task)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (can be local path or S3 path like s3://bucket/prefix)",
    )

    # Processing parameters - Multi-model mode only
    parser.add_argument("--classifier-pkl", help="Classifier pickle file for filtering")
    parser.add_argument("--classifier-threshold", type=float, default=0.75)
    parser.add_argument("--prefilter-model-type", help="Optional prefilter model type (e.g., 'CTRANSPATH')")
    parser.add_argument("--models", help="Comma-separated list of tile encoder models (e.g., 'UNI2,VIRCHOW2,CONCH1_5'). Can also include slide models (GIGAPATH_SLIDE, TITAN_SLIDE) which will be automatically categorized.")
    parser.add_argument("--slide-models", help="Comma-separated list of slide models for aggregation (e.g., 'TITAN_SLIDE,GIGAPATH_SLIDE')")
    parser.add_argument("--aggregation-method", default="model", choices=["model"], help="Aggregation method (only 'model' supported)")


    # SegConfig parameters
    parser.add_argument(
        "--seg-config-group",
        choices=["default", "biopsy", "resection", "tcga"],
        help="SegConfig group preset (default, biopsy, resection, tcga). Overrides individual seg_config parameters.",
    )
    parser.add_argument(
        "--segment-threshold",
        type=int,
        help="Tissue segmentation threshold (default: 20 for default, varies by group)",
    )
    parser.add_argument(
        "--patch-size", type=int, help="Patch size in pixels (default: 256)"
    )
    parser.add_argument(
        "--step-size",
        type=int,
        help="Step size for patch extraction (default: same as patch_size)",
    )
    parser.add_argument("--mpp", type=float, help="Microns per pixel (default: 0.5)")
    parser.add_argument(
        "--seg-level",
        type=int,
        help="Segmentation pyramid level (default: -1 for auto)",
    )
    parser.add_argument(
        "--segment-max-value",
        type=int,
        help="Maximum pixel value for segmentation (default: 255)",
    )
    parser.add_argument(
        "--median-blur-ksize",
        type=int,
        help="Median blur kernel size (default: 7, varies by group)",
    )
    parser.add_argument(
        "--morphology-ex-kernel",
        type=int,
        help="Morphological closing kernel size (default: 0, varies by group)",
    )
    parser.add_argument(
        "--ref-patch-size",
        type=int,
        help="Reference patch size for thresholding (default: 512)",
    )
    parser.add_argument(
        "--use-otsu", action="store_true", help="Use Otsu thresholding (default: False)"
    )
    parser.add_argument(
        "--tissue-area-threshold",
        type=int,
        help="Tissue area threshold (default: 100, varies by group)",
    )
    parser.add_argument(
        "--hole-area-threshold",
        type=int,
        help="Hole area threshold (default: 16, varies by group)",
    )
    parser.add_argument(
        "--max-num-holes",
        type=int,
        help="Maximum number of holes (default: 8, varies by group)",
    )

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256, help="Tile batch size for feature extraction (default: 256). Lower this if you get OOM errors.")
    parser.add_argument(
        "--slide-batch-size",
        type=int,
        default=8,
        help="Slides per batch during slide-level aggregation (default: 8)",
    )
    parser.add_argument(
        "--distributed-slide-batch-size",
        type=int,
        default=None,
        help="Number of slides to group per distributed task for batch processing optimization (default: auto). "
        "When not specified (auto), automatically groups slides into batches of 8 for efficiency. "
        "Set to 1 to explicitly disable batching (one slide per task). "
        "Groups slides together to load models once instead of N times. "
        "Recommended: 8-16 for better efficiency with multi-slide processing.",
    )
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", action="store_false", dest="use_gpu")

    # SLURM resource requirements
    parser.add_argument(
        "--partition",
        default=None,
        help="SLURM partition (if not specified, uses SLURM default)",
    )
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument(
        "--mem", default="16G", help="Memory per task (e.g., 16G, 32GB)"
    )
    parser.add_argument("--time", default="02:00:00", help="Time limit (HH:MM:SS)")
    parser.add_argument("--gres", help="Generic resources (e.g., gpu:1)")
    parser.add_argument("--qos", help="Quality of service")

    # Array job options (deprecated)
    parser.add_argument(
        "--no-array",
        action="store_true",
        help="(Deprecated - job arrays removed) This flag is ignored. Individual job submission is now the default.",
    )

    # AWS S3 credentials
    parser.add_argument("--aws-access-key-id", help="AWS access key ID")
    parser.add_argument("--aws-secret-access-key", help="AWS secret access key")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument(
        "--aws-endpoint-url", help="Custom S3 endpoint URL (e.g., for MinIO or Ceph)"
    )

    # HuggingFace token
    parser.add_argument("--hf-token", help="HuggingFace token for gated models")

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
        help="Shared filesystem directory to cache models (default: ./model_cache)",
    )


    # Docker support
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Run tessellate-extract-features in Docker container",
    )
    parser.add_argument(
        "--docker-image",
        default="ghcr.io/biomedia-mira/mussel:latest",
        help="Docker image to use (default: ghcr.io/biomedia-mira/mussel:latest)",
    )
    parser.add_argument(
        "--docker-runtime",
        default="nvidia",
        help="Docker runtime (default: nvidia for GPU support)",
    )
    parser.add_argument(
        "--container-runtime",
        default="singularity",
        choices=["docker", "singularity", "apptainer"],
        help="Container runtime to use (default: singularity)",
    )
    parser.add_argument(
        "--mount-code-dir",
        help="Local code directory to mount into container (e.g., for development)",
    )

    # Submission
    parser.add_argument(
        "--submit", action="store_true", help="Actually submit to SLURM"
    )

    args = parser.parse_args()

    # configargparse handles config file merging automatically
    config_defaults = {}

    # For backward compatibility with custom config loader
    if hasattr(args, "config") and args.config and load_config_defaults:
        try:
            config_defaults = load_config_defaults(args.config, backend="slurm")
        except Exception as e:
            print(f"WARNING: Failed to load config file with custom loader: {e}")
            config_defaults = {}

    # Pre-download models if requested and using batch processing
    model_paths = {}
    if args.pre_download_models and args.csv_manifest:
        print("\n[Model Pre-Download] Starting model pre-download process...")

        # Determine which models need to be downloaded
        models_to_download = []

        # Add prefilter model if specified
        prefilter_model_type = args.prefilter_model_type or config_defaults.get(
            "prefilter_model_type"
        )
        if prefilter_model_type:
            models_to_download.append(prefilter_model_type)

        # Add tile encoder models (required for multi-model mode)
        models_arg = args.models or config_defaults.get("models") or config_defaults.get("model_types")
        if models_arg:
            # Multiple models specified (comma-separated)
            model_list = [m.strip() for m in models_arg.split(",")]
            models_to_download.extend(model_list)

        # Add slide models (optional)
        slide_models = args.slide_models or config_defaults.get("slide_models") or config_defaults.get("slide_model_types")
        if slide_models:
            # Multiple slide models specified (comma-separated)
            slide_model_list = [m.strip() for m in slide_models.split(",")]
            models_to_download.extend(slide_model_list)

        # Remove duplicates
        models_to_download = list(set(models_to_download))

        if models_to_download:
            print(
                f"[Model Pre-Download] Models to download: {', '.join(models_to_download)}"
            )

            try:
                # Use save_model CLI to download models to model_dir
                import subprocess

                # Build model_types argument as a list string for Hydra
                model_types_str = "[" + ",".join(models_to_download) + "]"

                # Use docker/apptainer if requested
                if args.use_docker:
                    runtime = args.container_runtime
                    if runtime in ("singularity", "apptainer"):
                        # Use singularity/apptainer
                        print(
                            f"[Model Pre-Download] Using {runtime} for model download"
                        )

                        # Format image for singularity
                        docker_image = args.docker_image
                        # Check if it's a local SIF file
                        is_local_sif = docker_image.endswith(".sif") or os.path.isfile(
                            docker_image
                        )
                        if not is_local_sif and not docker_image.startswith(
                            "docker://"
                        ):
                            docker_image = f"docker://{docker_image}"

                        # Build environment variables
                        env_args = []
                        if args.hf_token:
                            env_args.extend(["--env", f"HF_TOKEN={args.hf_token}"])

                        # Set HuggingFace cache dir to use the mounted model_cache
                        env_args.extend(
                            [
                                "--env",
                                "HF_HOME=/workspace/model_cache/.cache/huggingface",
                            ]
                        )
                        env_args.extend(
                            [
                                "--env",
                                "TRANSFORMERS_CACHE=/workspace/model_cache/.cache/huggingface",
                            ]
                        )

                        # Build bind mounts
                        bind_args = [
                            "--bind",
                            f"{os.path.abspath(args.model_dir)}:/workspace/model_cache",
                            "--bind",
                            f"{os.path.expanduser('~/.cache')}:{os.path.expanduser('~/.cache')}",
                        ]

                        cmd = (
                            [
                                runtime,
                                "exec",
                                "--nv",  # Enable GPU support
                            ]
                            + env_args
                            + bind_args
                            + [
                                docker_image,
                                "python",
                                "-m",
                                "mussel.cli.save_model",
                                f"model_types={model_types_str}",
                                "model_dir=/workspace/model_cache",
                            ]
                        )
                    else:
                        # Use docker
                        print(f"[Model Pre-Download] Using docker for model download")

                        # Build environment variables
                        env_args = []
                        if args.hf_token:
                            env_args.extend(["-e", f"HF_TOKEN={args.hf_token}"])

                        # Set HuggingFace cache dir to use the mounted model_cache
                        env_args.extend(
                            ["-e", "HF_HOME=/workspace/model_cache/.cache/huggingface"]
                        )
                        env_args.extend(
                            [
                                "-e",
                                "TRANSFORMERS_CACHE=/workspace/model_cache/.cache/huggingface",
                            ]
                        )

                        cmd = (
                            [
                                "docker",
                                "run",
                                "--rm",
                                f"--runtime={args.docker_runtime}",
                            ]
                            + env_args
                            + [
                                "-v",
                                f"{os.path.abspath(args.model_dir)}:/workspace/model_cache",
                                args.docker_image,
                                "python",
                                "-m",
                                "mussel.cli.save_model",
                                f"model_types={model_types_str}",
                                "model_dir=/workspace/model_cache",
                            ]
                        )
                else:
                    # Use direct uv run
                    cmd = [
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "mussel.cli.save_model",
                        f"model_types={model_types_str}",
                        f"model_dir={args.model_dir}",
                    ]

                print(f"[Model Pre-Download] Running: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent.parent.parent,
                )

                if result.returncode != 0:
                    print(
                        f"[Model Pre-Download] Some models may not have been cached successfully"
                    )
                    print(f"[Model Pre-Download] Exit code: {result.returncode}")
                    if result.stdout:
                        print(f"[Model Pre-Download] Output:\n{result.stdout}")
                    if result.stderr and "GIGAPATH" not in result.stderr:
                        print(f"[Model Pre-Download] Errors:\n{result.stderr}")
                    print(f"[Model Pre-Download] Continuing with job submission")
                    print(
                        f"[Model Pre-Download] Tasks will download any missing models from HuggingFace Hub"
                    )
                else:
                    print(result.stdout)
                    print(
                        f"[Model Pre-Download] ✓ Models successfully cached to: {args.model_dir}"
                    )

            except Exception as e:
                print(f"[Model Pre-Download] Error during pre-download: {e}")
                print(f"[Model Pre-Download] Continuing with job submission")
                print(
                    f"[Model Pre-Download] Tasks will download models from HuggingFace Hub"
                )
        else:
            print(
                "[Model Pre-Download] All models provided by user, skipping pre-download"
            )

    # Apply user-provided model paths (override pre-downloaded if both specified)
    # Note: In multi-model mode, we don't need to track individual model paths
    # The model_dir will contain all models
    
    # Validate CTRANSPATH configuration
    # CTRANSPATH requires model to be pre-downloaded to model_dir
    if args.csv_manifest:
        prefilter_model = args.prefilter_model_type or config_defaults.get(
            "prefilter_model_type"
        )
        
        if prefilter_model and prefilter_model.upper() == "CTRANSPATH":
            print("\n⚠️  WARNING: CTRANSPATH model must be pre-downloaded to model_dir")
            print("   CTRANSPATH does not have a default HuggingFace path and cannot be automatically downloaded.")
            print("   Ensure the model is available in the model_dir before submitting tasks.\n")

    # Validate single task arguments
    if args.job_name:
        if not args.slide_path or not args.output_dir:
            parser.error("--job-name requires --slide-path and --output-dir")

    # Create submitter
    submitter = SlurmJobSubmitter()

    # Submit tasks
    if args.job_name:
        # Single task submission (multi-model mode)
        submitter.submit_task(
            job_name=args.job_name,
            slide_path=args.slide_path,
            output_dir=args.output_dir,
            classifier_pkl=args.classifier_pkl,
            classifier_threshold=args.classifier_threshold,
            prefilter_model_type=args.prefilter_model_type,
            model_types=args.models or config_defaults.get("models") or config_defaults.get("model_types"),
            aggregation_method=args.aggregation_method,
            slide_model_types=args.slide_models
            or config_defaults.get("slide_models") or config_defaults.get("slide_model_types"),
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
            aws_endpoint_url=args.aws_endpoint_url,
            hf_token=args.hf_token,
            use_docker=args.use_docker,
            docker_image=args.docker_image,
            docker_runtime=args.docker_runtime,
            container_runtime=args.container_runtime,
            mount_code_dir=args.mount_code_dir,
            model_dir=args.model_dir,
            submit=args.submit,
        )
    else:
        # CSV manifest - multi-model mode
        # Categorize models into tile vs slide encoders
        SLIDE_ENCODERS = {'GIGAPATH_SLIDE', 'TITAN_SLIDE'}
        model_types_input = args.models or config_defaults.get("models") or config_defaults.get("model_types")
        slide_models_input = args.slide_models or config_defaults.get("slide_models") or config_defaults.get("slide_model_types")
        
        # Parse model lists
        if model_types_input:
            if isinstance(model_types_input, str):
                model_types_list = [m.strip() for m in model_types_input.split(',')]
            elif isinstance(model_types_input, list):
                model_types_list = model_types_input
            else:
                model_types_list = []
        else:
            model_types_list = []
            
        if slide_models_input:
            if isinstance(slide_models_input, str):
                slide_models_list = [m.strip() for m in slide_models_input.split(',')]
            elif isinstance(slide_models_input, list):
                slide_models_list = slide_models_input
            else:
                slide_models_list = []
        else:
            slide_models_list = []
        
        # Separate tile encoders and slide encoders from model_types
        tile_models = [m for m in model_types_list if m and m not in SLIDE_ENCODERS]
        slide_models_from_models = [m for m in model_types_list if m and m in SLIDE_ENCODERS]
        
        # Combine slide models from both sources
        all_slide_models = list(set(slide_models_from_models + slide_models_list))
        
        # Convert back to comma-separated strings or None
        model_types_str = ','.join(tile_models) if tile_models else None
        slide_model_types_str = ','.join(all_slide_models) if all_slide_models else None
        
        # Validate that at least one model type is specified
        if not model_types_str and not slide_model_types_str:
            parser.error("At least one of --models or --slide-models must be specified with --csv-manifest")
        
        csv_kwargs = {
            "classifier_pkl": args.classifier_pkl,
            "classifier_threshold": args.classifier_threshold,
            "prefilter_model_type": args.prefilter_model_type,
            "model_types": model_types_str,
            "aggregation_method": args.aggregation_method,
            "slide_model_types": slide_model_types_str,
            "seg_config_group": args.seg_config_group,
            "segment_threshold": args.segment_threshold,
            "patch_size": args.patch_size,
            "step_size": args.step_size,
            "mpp": args.mpp,
            "seg_level": args.seg_level,
            "segment_max_value": args.segment_max_value,
            "median_blur_ksize": args.median_blur_ksize,
            "morphology_ex_kernel": args.morphology_ex_kernel,
            "ref_patch_size": args.ref_patch_size,
            "use_otsu": args.use_otsu,
            "tissue_area_threshold": args.tissue_area_threshold,
            "hole_area_threshold": args.hole_area_threshold,
            "max_num_holes": args.max_num_holes,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "slide_batch_size": args.slide_batch_size,
            "use_gpu": args.use_gpu,
            "partition": args.partition,
            "cpus_per_task": args.cpus_per_task,
            "mem": args.mem,
            "time": args.time,
            "gres": args.gres,
            "qos": args.qos,
            "aws_access_key_id": args.aws_access_key_id,
            "aws_secret_access_key": args.aws_secret_access_key,
            "aws_region": args.aws_region,
            "aws_endpoint_url": args.aws_endpoint_url,
            "hf_token": args.hf_token,
            "use_docker": args.use_docker,
            "docker_image": args.docker_image,
            "docker_runtime": args.docker_runtime,
            "container_runtime": args.container_runtime,
            "mount_code_dir": args.mount_code_dir,
            "model_dir": args.model_dir,
            "submit": args.submit,
        }

        # Remove None values to avoid overriding config file defaults with None
        csv_kwargs = {k: v for k, v in csv_kwargs.items() if v is not None}

        submitter.submit_tasks_from_csv(
            csv_file=args.csv_manifest,
            output_dir=args.output_dir,
            distributed_slide_batch_size=args.distributed_slide_batch_size,
            **csv_kwargs,
        )


if __name__ == "__main__":
    main()
