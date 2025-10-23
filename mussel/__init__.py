"""Mussel - Computational Pathology Toolkit for Whole-Slide Images

Mussel provides tools for processing whole-slide images (WSI) in computational pathology,
including tiling, feature extraction using foundation models, and zero-shot classification.

Key Features:
    - Tissue segmentation and tiling with adaptive resolution
    - Feature extraction using multiple pathology foundation models
    - Zero-shot tissue classification with natural language descriptions
    - Efficient tile caching for training workflows
    - Multi-GPU support for large-scale processing

Main Components:
    - CLI tools: Command-line utilities for WSI processing
    - Models: Foundation model wrappers and factory
    - Utils: Utility functions for segmentation and feature extraction
    - Datasets: Data loading and preprocessing utilities

Example:
    Basic workflow for processing a slide:
    
    >>> # Tile the slide
    >>> tessellate slide_path=slide.svs output_h5_path=tiles.h5
    
    >>> # Extract features
    >>> extract_features slide_path=slide.svs patch_h5_path=tiles.h5 \
    ...     output_h5_path=features.h5 output_pt_path=features.pt
    
    >>> # Annotate tiles
    >>> create_class_embeddings classes='["tumor","stroma"]' \
    ...     output_pt_path=classes.pt
    >>> annotate features_pt_path=features.pt \
    ...     class_embedding_pt_path=classes.pt \
    ...     classes='["tumor","stroma"]' output_csv_path=annotations.csv

For more information, see README.md and README-commands.md.

License:
    GPL v3.0 - See LICENSE.md for details.
    Forked from CLAM © Mahmood Lab
"""

__version__ = "1.1.1"
__author__ = "Pathology Data Mining Team"
__license__ = "GPL-3.0"
