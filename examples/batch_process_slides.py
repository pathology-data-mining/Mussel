#!/usr/bin/env python3
"""
Example: Programmatic batch processing of whole-slide images

This script demonstrates how to use the batch processing functionality
programmatically in Python code.
"""

from pathlib import Path
from omegaconf import OmegaConf

from mussel.cli.tessellate_extract_features import (
    TessellateExtractFeaturesConfig,
    main,
)
from mussel.cli.tessellate import SegConfig
from mussel.models import ModelType


def batch_process_slides_example():
    """
    Example of batch processing multiple slides with slide-level aggregation.
    """
    # Configuration
    slides_dir = Path("/path/to/slides")
    output_dir = Path("./batch_output")
    
    # Find all slide files
    slide_paths = list(slides_dir.glob("*.svs"))
    
    if not slide_paths:
        print(f"No slides found in {slides_dir}")
        return
    
    print(f"Found {len(slide_paths)} slides")
    
    # Create configuration
    seg_config = SegConfig(
        segment_threshold=0,  # Default tissue segmentation
        patch_size=256,
        patch_level=0,
        mpp=None,
    )
    
    cfg = TessellateExtractFeaturesConfig(
        # Input slides
        slide_paths=[str(p) for p in slide_paths],
        slide_ids=None,  # Auto-generate from filenames
        
        # Output configuration
        output_dir=str(output_dir),
        output_h5_suffix="features.h5",
        output_pt_suffix="features.pt",
        
        # Model configuration for slide-level aggregation
        aggregation_method="model",
        slide_model_type=ModelType.GIGAPATH_SLIDE,
        slide_batch_size=8,  # Process 8 slides at a time
        
        # Patch encoder configuration
        prefilter_model_type=ModelType.GIGAPATH,  # Auto-inferred from GIGAPATH_SLIDE
        
        # Processing configuration
        seg_config=seg_config,
        num_workers=8,
        batch_size=128,
        use_gpu=True,
        gpu_device_id=0,
        
        # Optional: Save intermediate files for debugging
        keep_intermediate_files=False,
        save_features_to_h5=True,
    )
    
    # Run batch processing
    print("Starting batch processing...")
    main(OmegaConf.create(cfg))
    print("Batch processing complete!")
    
    # Output files are organized by slide ID
    print(f"\nOutput files in: {output_dir}")
    for slide_path in slide_paths:
        slide_id = slide_path.stem
        h5_file = output_dir / f"{slide_id}.features.h5"
        pt_file = output_dir / f"{slide_id}.features.pt"
        print(f"  {slide_id}:")
        print(f"    - {h5_file}")
        print(f"    - {pt_file}")


def batch_process_with_filtering_example():
    """
    Example of batch processing with tile filtering.
    """
    slide_paths = [
        "/path/to/slide1.svs",
        "/path/to/slide2.svs",
        "/path/to/slide3.svs",
    ]
    
    classifier_pkl = "/path/to/classifier.pkl"
    output_dir = Path("./filtered_output")
    
    seg_config = SegConfig(segment_threshold=0)
    
    cfg = TessellateExtractFeaturesConfig(
        slide_paths=slide_paths,
        output_dir=str(output_dir),
        
        # Enable filtering
        classifier_pkl=classifier_pkl,
        classifier_threshold=0.75,
        
        # Use same model for pre and post filtering (optimized)
        prefilter_model_type=ModelType.CTRANSPATH,
        postfilter_model_type=None,  # Will use prefilter_model_type
        
        seg_config=seg_config,
        num_workers=4,
        batch_size=64,
        use_gpu=True,
    )
    
    print("Starting batch processing with filtering...")
    main(OmegaConf.create(cfg))
    print("Complete!")


def simple_batch_example():
    """
    Simplest example - just process multiple slides without aggregation.
    """
    slide_paths = [
        "/path/to/slide1.svs",
        "/path/to/slide2.svs",
    ]
    
    cfg = TessellateExtractFeaturesConfig(
        slide_paths=slide_paths,
        output_dir="./simple_output",
        prefilter_model_type=ModelType.RESNET50,
        seg_config=SegConfig(segment_threshold=0),
        num_workers=4,
        batch_size=64,
        use_gpu=False,  # Use CPU
    )
    
    main(OmegaConf.create(cfg))


if __name__ == "__main__":
    # Run the main example
    batch_process_slides_example()
    
    # Uncomment to try other examples:
    # batch_process_with_filtering_example()
    # simple_batch_example()
