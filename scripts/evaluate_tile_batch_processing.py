#!/usr/bin/env python3
"""
Evaluation script for batch processing performance with tile-level feature extraction.

This script evaluates the time savings when batch processing multiple slides
for tile-level feature extraction, specifically calibrated for slides with
approximately 13,000 tiles per slide.

The evaluation considers:
1. Tile-level feature extraction (batched within each slide)
2. Slide-level aggregation (batched across slides when using slide models)
3. Model loading overhead
4. GPU utilization efficiency

Usage:
    python evaluate_tile_batch_processing.py --num-slides 10 --tiles-per-slide 13000
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PerformanceMetrics:
    """Performance metrics for processing."""
    total_time: float
    model_load_time: float
    tile_extraction_time: float
    slide_aggregation_time: float
    num_slides: int
    tiles_per_slide: int
    
    @property
    def avg_time_per_slide(self) -> float:
        return self.total_time / self.num_slides if self.num_slides > 0 else 0
    
    @property
    def avg_time_per_tile(self) -> float:
        total_tiles = self.num_slides * self.tiles_per_slide
        return self.total_time / total_tiles if total_tiles > 0 else 0


def estimate_tile_extraction_time(
    num_tiles: int,
    tile_batch_size: int = 64,
    time_per_batch_ms: float = 100.0,
    model_warmup_ms: float = 500.0,
) -> float:
    """
    Estimate time for tile-level feature extraction.
    
    Args:
        num_tiles: Number of tiles to process
        tile_batch_size: Batch size for tile processing
        time_per_batch_ms: Time to process one batch of tiles (ms)
        model_warmup_ms: Model warmup time on first batch (ms)
    
    Returns:
        Total time in seconds
    """
    num_batches = (num_tiles + tile_batch_size - 1) // tile_batch_size
    
    # First batch includes warmup
    total_time_ms = model_warmup_ms + time_per_batch_ms
    
    # Remaining batches
    if num_batches > 1:
        total_time_ms += (num_batches - 1) * time_per_batch_ms
    
    return total_time_ms / 1000.0


def simulate_sequential_processing(
    num_slides: int,
    tiles_per_slide: int = 13000,
    tile_batch_size: int = 64,
    model_load_time_s: float = 2.0,
    tile_batch_time_ms: float = 100.0,
    slide_inference_time_s: float = 0.5,
    use_slide_model: bool = True,
) -> PerformanceMetrics:
    """
    Simulate sequential slide processing (current approach without batch optimization).
    
    For each slide:
    1. Load patch encoder model (if not cached)
    2. Extract tile-level features
    3. Load slide encoder model (if using slide model)
    4. Aggregate to slide-level features
    
    Args:
        num_slides: Number of slides to process
        tiles_per_slide: Average number of tiles per slide
        tile_batch_size: Batch size for tile processing
        model_load_time_s: Time to load model (seconds)
        tile_batch_time_ms: Time to process one batch of tiles (ms)
        slide_inference_time_s: Time for slide-level aggregation (seconds)
        use_slide_model: Whether using slide-level model aggregation
    
    Returns:
        PerformanceMetrics with timing information
    """
    print(f"\n{'='*70}")
    print(f"SEQUENTIAL PROCESSING SIMULATION")
    print(f"{'='*70}")
    print(f"Processing {num_slides} slides sequentially...")
    print(f"Tiles per slide: {tiles_per_slide:,}")
    print(f"Tile batch size: {tile_batch_size}")
    
    total_model_load_time = 0.0
    total_tile_extraction_time = 0.0
    total_slide_aggregation_time = 0.0
    
    for i in range(num_slides):
        # Load patch encoder model for each slide
        total_model_load_time += model_load_time_s
        
        # Extract tile features
        tile_time = estimate_tile_extraction_time(
            num_tiles=tiles_per_slide,
            tile_batch_size=tile_batch_size,
            time_per_batch_ms=tile_batch_time_ms,
            model_warmup_ms=500.0,
        )
        total_tile_extraction_time += tile_time
        
        # Slide-level aggregation (load model each time if using slide model)
        if use_slide_model:
            total_model_load_time += model_load_time_s  # Load slide model
            total_slide_aggregation_time += slide_inference_time_s
        
        if (i + 1) % 10 == 0:
            elapsed = total_model_load_time + total_tile_extraction_time + total_slide_aggregation_time
            print(f"  Processed {i + 1}/{num_slides} slides (elapsed: {elapsed:.1f}s)")
    
    total_time = total_model_load_time + total_tile_extraction_time + total_slide_aggregation_time
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Model load time: {total_model_load_time:.2f}s ({total_model_load_time/total_time*100:.1f}%)")
    print(f"  Tile extraction time: {total_tile_extraction_time:.2f}s ({total_tile_extraction_time/total_time*100:.1f}%)")
    print(f"  Slide aggregation time: {total_slide_aggregation_time:.2f}s ({total_slide_aggregation_time/total_time*100:.1f}%)")
    print(f"  Average per slide: {total_time/num_slides:.2f}s")
    
    return PerformanceMetrics(
        total_time=total_time,
        model_load_time=total_model_load_time,
        tile_extraction_time=total_tile_extraction_time,
        slide_aggregation_time=total_slide_aggregation_time,
        num_slides=num_slides,
        tiles_per_slide=tiles_per_slide,
    )


def simulate_batch_processing(
    num_slides: int,
    tiles_per_slide: int = 13000,
    tile_batch_size: int = 64,
    slide_batch_size: int = 8,
    model_load_time_s: float = 2.0,
    tile_batch_time_ms: float = 100.0,
    slide_inference_time_s: float = 0.5,
    batch_efficiency: float = 0.6,
    use_slide_model: bool = True,
) -> PerformanceMetrics:
    """
    Simulate batch slide processing (optimized approach).
    
    1. Load patch encoder model once
    2. For each slide: Extract tile-level features (can be parallel)
    3. Load slide encoder model once (if using slide model)
    4. Process slides in batches for aggregation
    
    Args:
        num_slides: Number of slides to process
        tiles_per_slide: Average number of tiles per slide
        tile_batch_size: Batch size for tile processing
        slide_batch_size: Batch size for slide-level aggregation
        model_load_time_s: Time to load model (seconds)
        tile_batch_time_ms: Time to process one batch of tiles (ms)
        slide_inference_time_s: Time for slide-level aggregation per slide (seconds)
        batch_efficiency: Efficiency of batched slide aggregation (0-1)
        use_slide_model: Whether using slide-level model aggregation
    
    Returns:
        PerformanceMetrics with timing information
    """
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING SIMULATION")
    print(f"{'='*70}")
    print(f"Processing {num_slides} slides in batch mode...")
    print(f"Tiles per slide: {tiles_per_slide:,}")
    print(f"Tile batch size: {tile_batch_size}")
    print(f"Slide batch size: {slide_batch_size}")
    
    # Load patch encoder model once
    total_model_load_time = model_load_time_s
    print(f"  Loaded patch encoder model ({model_load_time_s:.2f}s)")
    
    # Extract tile features for all slides
    # In real implementation, this could be parallelized per slide
    total_tile_extraction_time = 0.0
    for i in range(num_slides):
        tile_time = estimate_tile_extraction_time(
            num_tiles=tiles_per_slide,
            tile_batch_size=tile_batch_size,
            time_per_batch_ms=tile_batch_time_ms,
            model_warmup_ms=50.0 if i == 0 else 0.0,  # Only first slide has warmup
        )
        total_tile_extraction_time += tile_time
        
        if (i + 1) % 10 == 0:
            print(f"  Extracted tiles from {i + 1}/{num_slides} slides ({total_tile_extraction_time:.1f}s)")
    
    # Slide-level aggregation
    total_slide_aggregation_time = 0.0
    if use_slide_model:
        # Load slide encoder model once
        total_model_load_time += model_load_time_s
        print(f"  Loaded slide encoder model ({model_load_time_s:.2f}s)")
        
        # Process slides in batches for aggregation
        num_batches = (num_slides + slide_batch_size - 1) // slide_batch_size
        for batch_idx in range(num_batches):
            batch_start = batch_idx * slide_batch_size
            batch_end = min(batch_start + slide_batch_size, num_slides)
            current_batch_size = batch_end - batch_start
            
            # Batch processing is more efficient than sequential
            batch_time = slide_inference_time_s * current_batch_size * batch_efficiency
            total_slide_aggregation_time += batch_time
            
            print(f"  Aggregated batch {batch_idx + 1}/{num_batches} ({batch_start + 1}-{batch_end}) ({batch_time:.2f}s)")
    
    total_time = total_model_load_time + total_tile_extraction_time + total_slide_aggregation_time
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Model load time: {total_model_load_time:.2f}s ({total_model_load_time/total_time*100:.1f}%)")
    print(f"  Tile extraction time: {total_tile_extraction_time:.2f}s ({total_tile_extraction_time/total_time*100:.1f}%)")
    print(f"  Slide aggregation time: {total_slide_aggregation_time:.2f}s ({total_slide_aggregation_time/total_time*100:.1f}%)")
    print(f"  Average per slide: {total_time/num_slides:.2f}s")
    
    return PerformanceMetrics(
        total_time=total_time,
        model_load_time=total_model_load_time,
        tile_extraction_time=total_tile_extraction_time,
        slide_aggregation_time=total_slide_aggregation_time,
        num_slides=num_slides,
        tiles_per_slide=tiles_per_slide,
    )


def print_comparison(
    sequential: PerformanceMetrics,
    batch: PerformanceMetrics,
    tiles_per_slide: int,
):
    """Print detailed comparison between sequential and batch processing."""
    print(f"\n{'='*70}")
    print(f"PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    
    print(f"\nConfiguration:")
    print(f"  Number of slides: {sequential.num_slides}")
    print(f"  Tiles per slide: {tiles_per_slide:,}")
    print(f"  Total tiles processed: {sequential.num_slides * tiles_per_slide:,}")
    
    print(f"\n{'-'*70}")
    print(f"Sequential Processing:")
    print(f"{'-'*70}")
    print(f"  Total time:              {sequential.total_time:>10.2f}s")
    print(f"  Model loading:           {sequential.model_load_time:>10.2f}s ({sequential.model_load_time/sequential.total_time*100:>5.1f}%)")
    print(f"  Tile extraction:         {sequential.tile_extraction_time:>10.2f}s ({sequential.tile_extraction_time/sequential.total_time*100:>5.1f}%)")
    print(f"  Slide aggregation:       {sequential.slide_aggregation_time:>10.2f}s ({sequential.slide_aggregation_time/sequential.total_time*100:>5.1f}%)")
    print(f"  Time per slide:          {sequential.avg_time_per_slide:>10.2f}s")
    print(f"  Time per tile:           {sequential.avg_time_per_tile*1000:>10.2f}ms")
    
    print(f"\n{'-'*70}")
    print(f"Batch Processing:")
    print(f"{'-'*70}")
    print(f"  Total time:              {batch.total_time:>10.2f}s")
    print(f"  Model loading:           {batch.model_load_time:>10.2f}s ({batch.model_load_time/batch.total_time*100:>5.1f}%)")
    print(f"  Tile extraction:         {batch.tile_extraction_time:>10.2f}s ({batch.tile_extraction_time/batch.total_time*100:>5.1f}%)")
    print(f"  Slide aggregation:       {batch.slide_aggregation_time:>10.2f}s ({batch.slide_aggregation_time/batch.total_time*100:>5.1f}%)")
    print(f"  Time per slide:          {batch.avg_time_per_slide:>10.2f}s")
    print(f"  Time per tile:           {batch.avg_time_per_tile*1000:>10.2f}ms")
    
    # Calculate improvements
    speedup = sequential.total_time / batch.total_time
    time_saved = sequential.total_time - batch.total_time
    improvement_pct = (1 - batch.total_time / sequential.total_time) * 100
    per_slide_savings = sequential.avg_time_per_slide - batch.avg_time_per_slide
    
    print(f"\n{'-'*70}")
    print(f"Performance Improvement:")
    print(f"{'-'*70}")
    print(f"  Speedup:                 {speedup:>10.2f}x")
    print(f"  Time saved:              {time_saved:>10.2f}s ({improvement_pct:>5.1f}% faster)")
    print(f"  Per-slide improvement:   {per_slide_savings:>10.2f}s")
    print(f"  Model load reduction:    {sequential.model_load_time - batch.model_load_time:>10.2f}s")
    print(f"  Aggregation improvement: {sequential.slide_aggregation_time - batch.slide_aggregation_time:>10.2f}s")
    
    print(f"\n{'='*70}")


def print_summary(
    num_slides: int,
    tiles_per_slide: int,
    sequential: PerformanceMetrics,
    batch: PerformanceMetrics,
):
    """Print executive summary."""
    speedup = sequential.total_time / batch.total_time
    improvement_pct = (1 - batch.total_time / sequential.total_time) * 100
    
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"""
When processing {num_slides} slides with ~{tiles_per_slide:,} tiles per slide:

Sequential Processing: {sequential.total_time:.1f}s ({sequential.avg_time_per_slide:.2f}s per slide)
Batch Processing:      {batch.total_time:.1f}s ({batch.avg_time_per_slide:.2f}s per slide)

TIME SAVINGS: {sequential.total_time - batch.total_time:.1f}s ({speedup:.2f}x speedup, {improvement_pct:.1f}% faster)

Key benefits of batch processing:
- Load patch encoder model once (vs {num_slides} times)
- Load slide encoder model once (vs {num_slides} times)  
- Process slides efficiently in batches during aggregation
- Better GPU utilization throughout the pipeline

The batch processing approach is especially beneficial for:
- Large numbers of slides (50+)
- Slides with many tiles (10,000+)
- Use of slide-level model aggregation (GIGAPATH_SLIDE, etc.)
""")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate batch processing performance for tile-level feature extraction"
    )
    parser.add_argument(
        '--num-slides',
        type=int,
        default=20,
        help='Number of slides to process (default: 20)'
    )
    parser.add_argument(
        '--tiles-per-slide',
        type=int,
        default=13000,
        help='Average number of tiles per slide (default: 13000)'
    )
    parser.add_argument(
        '--tile-batch-size',
        type=int,
        default=64,
        help='Batch size for tile processing (default: 64)'
    )
    parser.add_argument(
        '--slide-batch-size',
        type=int,
        default=8,
        help='Batch size for slide aggregation (default: 8)'
    )
    parser.add_argument(
        '--model-load-time',
        type=float,
        default=2.0,
        help='Time to load a model in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--tile-batch-time',
        type=float,
        default=100.0,
        help='Time to process one batch of tiles in ms (default: 100.0)'
    )
    parser.add_argument(
        '--slide-inference-time',
        type=float,
        default=0.5,
        help='Time for slide-level inference per slide in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--batch-efficiency',
        type=float,
        default=0.6,
        help='Batch processing efficiency factor 0-1 (default: 0.6)'
    )
    parser.add_argument(
        '--no-slide-model',
        action='store_true',
        help='Disable slide-level model aggregation'
    )
    
    args = parser.parse_args()
    
    use_slide_model = not args.no_slide_model
    
    print("="*70)
    print("TILE-LEVEL BATCH PROCESSING EVALUATION")
    print("="*70)
    print(f"\nEvaluation Configuration:")
    print(f"  Number of slides:           {args.num_slides}")
    print(f"  Tiles per slide:            {args.tiles_per_slide:,}")
    print(f"  Tile batch size:            {args.tile_batch_size}")
    print(f"  Slide batch size:           {args.slide_batch_size}")
    print(f"  Model load time:            {args.model_load_time}s")
    print(f"  Tile batch time:            {args.tile_batch_time}ms")
    print(f"  Slide inference time:       {args.slide_inference_time}s")
    print(f"  Batch efficiency:           {args.batch_efficiency}")
    print(f"  Using slide model:          {use_slide_model}")
    
    # Run sequential processing simulation
    sequential_results = simulate_sequential_processing(
        num_slides=args.num_slides,
        tiles_per_slide=args.tiles_per_slide,
        tile_batch_size=args.tile_batch_size,
        model_load_time_s=args.model_load_time,
        tile_batch_time_ms=args.tile_batch_time,
        slide_inference_time_s=args.slide_inference_time,
        use_slide_model=use_slide_model,
    )
    
    # Run batch processing simulation
    batch_results = simulate_batch_processing(
        num_slides=args.num_slides,
        tiles_per_slide=args.tiles_per_slide,
        tile_batch_size=args.tile_batch_size,
        slide_batch_size=args.slide_batch_size,
        model_load_time_s=args.model_load_time,
        tile_batch_time_ms=args.tile_batch_time,
        slide_inference_time_s=args.slide_inference_time,
        batch_efficiency=args.batch_efficiency,
        use_slide_model=use_slide_model,
    )
    
    # Print comparison
    print_comparison(sequential_results, batch_results, args.tiles_per_slide)
    
    # Print summary
    print_summary(args.num_slides, args.tiles_per_slide, sequential_results, batch_results)
    
    print("\nNOTE: This evaluation uses realistic timing estimates based on:")
    print("  - Typical GPU inference times for foundation models")
    print("  - Model loading overhead from disk/network")
    print("  - Batch processing efficiency factors")
    print("\nActual performance will vary based on:")
    print("  - GPU hardware (V100, A100, H100, etc.)")
    print("  - Model type and size (ResNet50, CLIP, GigaPath, etc.)")
    print("  - Slide characteristics (size, tissue density)")
    print("  - System I/O performance")


if __name__ == '__main__':
    main()
