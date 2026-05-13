"""Common functionality for tessellate-extract-features workflows."""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import h5py
import tiffslide
from omegaconf import OmegaConf
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

from mussel.cli.tessellate import _build_artifact_remover
from mussel.utils import (filter_features, is_remote_path, load_classifier,
                          load_features_from_h5, safe_path_join, save_features,
                          save_hdf5, save_torch_tensor)
from mussel.utils.artifact_removal import GrandQCArtifactRemover
from mussel.utils.segment import (draw_slide_mask, save_patches_png,
                                  segment_tissue)


def _tessellate_and_filter(
    slide_path: str,
    slide_id: Optional[str],
    cfg,
    temp_dir: str,
    base_path: Union[str, Path],
    use_filtering: bool,
    prefilter_model_type,
    prefilter_model_path: Optional[str],
    skip_second_extraction: bool,
    output_mask_path: Optional[str] = None,
    artifact_remover_fn: Optional[GrandQCArtifactRemover] = None,
) -> Optional[dict]:
    """Tessellate a slide and optionally extract/filter features for tile selection.

    This is the shared core of both :func:`process_slide_tessellation_and_filtering`
    and :func:`process_slide_tessellation_only`.

    Args:
        slide_path: Path to the whole-slide image.
        slide_id: Optional slide identifier.
        cfg: Configuration object with seg_config, vis_config, etc.
        temp_dir: Temporary directory for intermediate files.
        base_path: Base path for output files.
        use_filtering: Whether to apply classifier-based tile filtering.
        prefilter_model_type: Model type used for pre-filter feature extraction.
        prefilter_model_path: Path to pre-filter model weights.
        skip_second_extraction: Whether the pre-filter model is also the final model
            (caller can use pre-filter features directly as final output).
        output_mask_path: Optional path to save a tissue mask visualisation.
        artifact_remover_fn: Pre-instantiated :class:`GrandQCArtifactRemover` to
            use for artifact removal.  When provided this instance is reused across
            slides (weights are loaded only once).  When ``None`` a new instance is
            created from ``cfg.seg_config`` flags, which incurs a model-weight
            download on first use for every slide.

    Returns:
        ``None`` on tessellation failure; otherwise a dict with:

        - ``tessellate_h5_path`` – HDF5 produced by tessellation.
        - ``final_coords_h5_path`` – HDF5 containing coords for the next extraction
          step (may be filtered coords, prefilter-features file, or tessellate file).
        - ``final_coords`` – numpy array of tile coordinates to process.
        - ``prefilter_features_h5_path`` – HDF5 of prefilter features, or ``None``.
        - ``filtered_features`` – torch.Tensor of filtered features when
          ``use_filtering and skip_second_extraction``, else ``None``.
        - ``original_coords_count`` – number of tiles before filtering.
    """
    # Step 1: Tessellate
    logger.info(f"Tessellating slide: {slide_path}")
    if cfg.keep_intermediate_files:
        tessellate_h5_path = safe_path_join(
            base_path, f"{Path(slide_path).stem}.tessellate.h5"
        )
    else:
        tessellate_h5_path = os.path.join(
            temp_dir, f"{Path(slide_path).stem}.tessellate.h5"
        )

    seg_cfg = OmegaConf.to_container(cfg.seg_config)

    if artifact_remover_fn is None:
        artifact_remover_fn = _build_artifact_remover(seg_cfg)

    # Strip config-only keys that are not segment_tissue() parameters.
    seg_cfg.pop("artifact_exclude_classes", None)

    if values := segment_tissue(
        slide_path=slide_path,
        slide_id=slide_id,
        output_h5_path=tessellate_h5_path,
        artifact_remover_fn=artifact_remover_fn,
        **seg_cfg,
    ):
        polygon, grid, coords, _ = values
    else:
        logger.error(f"Tessellation failed for {slide_path}")
        return None

    logger.info(f"Tessellation complete. Found {len(coords)} tiles.")

    if output_mask_path:
        mask = draw_slide_mask(
            slide_path,
            polygon,
            **OmegaConf.to_container(cfg.vis_config),
        )
        mask.save(output_mask_path)

    final_coords_h5_path = tessellate_h5_path
    final_coords = coords
    prefilter_features_h5_path = None
    filtered_features = None

    if use_filtering:
        logger.info(f"Extracting features for filtering: {slide_path}")
        if cfg.keep_intermediate_files:
            prefilter_features_h5_path = safe_path_join(
                base_path, f"{Path(slide_path).stem}.prefilter_features.h5"
            )
            prefilter_features_pt_path = safe_path_join(
                base_path, f"{Path(slide_path).stem}.prefilter_features.pt"
            )
        else:
            prefilter_features_h5_path = os.path.join(
                temp_dir, f"{Path(slide_path).stem}.prefilter_features.h5"
            )
            prefilter_features_pt_path = os.path.join(
                temp_dir, f"{Path(slide_path).stem}.prefilter_features.pt"
            )

        save_features(
            slide_path=slide_path,
            gpu_device_id=cfg.gpu_device_id,
            model_type=prefilter_model_type,
            model_path=prefilter_model_path,
            use_gpu=cfg.use_gpu,
            output_h5_path=prefilter_features_h5_path,
            output_pt_path=prefilter_features_pt_path,
            patch_h5_path=tessellate_h5_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            gpu_device_ids=cfg.gpu_device_ids,
        )

        logger.info(f"Filtering features: {slide_path}")
        classifier = load_classifier(cfg.classifier_pkl)
        features, all_coords = load_features_from_h5(
            prefilter_features_h5_path, prefilter_features_pt_path
        )
        filtered_features, filtered_coords = filter_features(
            features,
            all_coords,
            classifier,
            cfg.classifier_threshold,
        )
        logger.info(
            f"Filtering complete. {len(filtered_coords)} tiles passed (out of {len(coords)})."
        )

        if skip_second_extraction:
            # Pre-filter features are the final features; caller decides how to save them.
            final_coords = filtered_coords
            final_coords_h5_path = prefilter_features_h5_path
        else:
            if cfg.keep_intermediate_files:
                filtered_coords_h5_path = safe_path_join(
                    base_path, f"{Path(slide_path).stem}.filtered_coords.h5"
                )
            else:
                filtered_coords_h5_path = os.path.join(
                    temp_dir, f"{Path(slide_path).stem}.filtered_coords.h5"
                )

            save_hdf5(
                filtered_coords_h5_path,
                {"coords": filtered_coords},
                attr_h5_path=prefilter_features_h5_path,
                mode="w",
            )
            final_coords_h5_path = filtered_coords_h5_path
            final_coords = filtered_coords
            filtered_features = None  # not needed; coords saved above

    return {
        "tessellate_h5_path": tessellate_h5_path,
        "final_coords_h5_path": final_coords_h5_path,
        "final_coords": final_coords,
        "prefilter_features_h5_path": prefilter_features_h5_path,
        "filtered_features": filtered_features,
        "original_coords_count": len(coords),
    }


def process_slide_tessellation_and_filtering(
    slide_path: str,
    slide_id: Optional[str],
    output_h5_path: str,
    output_pt_path: str,
    cfg,
    temp_dir: str,
    base_path: Union[str, Path],
    use_filtering: bool,
    prefilter_model_type,
    prefilter_model_path: Optional[str],
    model_type,
    model_path: Optional[str],
    skip_second_extraction: bool,
    output_mask_path: Optional[str] = None,
    two_step_mode: bool = False,
    slide_model_path: Optional[str] = None,
    artifact_remover_fn: Optional[GrandQCArtifactRemover] = None,
) -> Optional[dict]:
    """Process a single slide through tessellation, optional filtering, and feature extraction.

    Args:
        slide_path: Path to the whole-slide image.
        slide_id: Optional slide identifier.
        output_h5_path: Path to save final HDF5 output.
        output_pt_path: Path to save final PyTorch output.
        cfg: Configuration object with seg_config, vis_config, etc.
        temp_dir: Temporary directory for intermediate files.
        base_path: Base path for output files.
        use_filtering: Whether to apply filtering.
        prefilter_model_type: Model type for pre-filtering.
        prefilter_model_path: Path to pre-filter model weights.
        model_type: Model type for post-filtering feature extraction.
        model_path: Path to post-filter model weights.
        skip_second_extraction: Whether to skip second extraction (when models are same).
        output_mask_path: Optional path to save mask visualization.
        two_step_mode: Whether using two-step aggregation (for batch processing).
        slide_model_path: Path to slide encoder model weights.
        artifact_remover_fn: Pre-instantiated artifact remover to reuse across slides.
            See :func:`_tessellate_and_filter` for details.

    Returns:
        ``None`` if processing failed (tessellation error); otherwise a dict with
        ``final_coords``, ``slide_path``, and ``tessellate_h5_path`` (plus any
        additional aggregation paths when a second extraction step is still needed).
    """
    result = _tessellate_and_filter(
        slide_path=slide_path,
        slide_id=slide_id,
        cfg=cfg,
        temp_dir=temp_dir,
        base_path=base_path,
        use_filtering=use_filtering,
        prefilter_model_type=prefilter_model_type,
        prefilter_model_path=prefilter_model_path,
        skip_second_extraction=skip_second_extraction,
        output_mask_path=output_mask_path,
        artifact_remover_fn=artifact_remover_fn,
    )
    if result is None:
        return None

    final_coords_h5_path = result["final_coords_h5_path"]
    final_coords = result["final_coords"]
    tessellate_h5_path = result["tessellate_h5_path"]
    filtered_features = result["filtered_features"]
    prefilter_features_h5_path = result["prefilter_features_h5_path"]

    if use_filtering and skip_second_extraction:
        # Pre-filter features are the final output; save them now.
        asset_dict = {"coords": final_coords}
        if cfg.save_features_to_h5:
            asset_dict["features"] = filtered_features.numpy()
        save_hdf5(
            output_h5_path,
            asset_dict,
            attr_h5_path=prefilter_features_h5_path,
            mode="w",
        )
        save_torch_tensor(output_pt_path, filtered_features)
        # Return tessellation info so the caller can still create visualizations.
        return {
            "final_coords": final_coords,
            "slide_path": slide_path,
            "tessellate_h5_path": tessellate_h5_path,
        }

    # Final extraction step
    if two_step_mode and cfg.aggregation_method != "identity":
        logger.info(f"Extracting patch features: {slide_path}")
        intermediate_h5_path = safe_path_join(
            base_path, f"{Path(slide_path).stem}.patch.h5"
        )
        save_features(
            slide_path=slide_path,
            gpu_device_id=cfg.gpu_device_id,
            model_type=model_type,
            model_path=model_path,
            use_gpu=cfg.use_gpu,
            output_h5_path=intermediate_h5_path,
            output_pt_path=None,
            patch_h5_path=final_coords_h5_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            gpu_device_ids=cfg.gpu_device_ids,
            intermediate_h5_path=None,
            aggregation_method="identity",
            slide_model_type=None,
            slide_model_path=None,
        )
        return {
            "intermediate_h5_path": intermediate_h5_path,
            "output_h5_path": output_h5_path,
            "output_pt_path": output_pt_path,
            "final_coords": final_coords,
            "slide_path": slide_path,
            "tessellate_h5_path": tessellate_h5_path,
        }
    else:
        logger.info(f"Extracting features: {slide_path}")
        save_features(
            slide_path=slide_path,
            gpu_device_id=cfg.gpu_device_id,
            model_type=model_type,
            model_path=model_path,
            use_gpu=cfg.use_gpu,
            output_h5_path=output_h5_path,
            output_pt_path=output_pt_path,
            patch_h5_path=final_coords_h5_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            gpu_device_ids=cfg.gpu_device_ids,
            intermediate_h5_path=getattr(cfg, "intermediate_h5_path", None),
            aggregation_method=cfg.aggregation_method,
            slide_model_type=getattr(cfg, "slide_model_type", None),
            slide_model_path=slide_model_path,
        )
        return {
            "final_coords": final_coords,
            "slide_path": slide_path,
            "tessellate_h5_path": tessellate_h5_path,
        }


def process_slide_tessellation_only(
    slide_path: str,
    slide_id: Optional[str],
    cfg,
    temp_dir: str,
    base_path: Union[str, Path],
    use_filtering: bool,
    prefilter_model_type,
    prefilter_model_path: Optional[str],
    skip_second_extraction: bool,
    output_mask_path: Optional[str] = None,
    artifact_remover_fn: Optional[GrandQCArtifactRemover] = None,
) -> Optional[dict]:
    """Process a slide through tessellation and optional filtering (no feature extraction).

    Used in batch mode to prepare slides for deferred batch feature extraction.

    Args:
        slide_path: Path to the whole-slide image.
        slide_id: Optional slide identifier.
        cfg: Configuration object with seg_config, vis_config, etc.
        temp_dir: Temporary directory for intermediate files.
        base_path: Base path for output files.
        use_filtering: Whether to apply filtering.
        prefilter_model_type: Model type for pre-filtering.
        prefilter_model_path: Path to pre-filter model weights.
        skip_second_extraction: Whether the pre-filter model is also the final model.
        output_mask_path: Optional path to save mask visualization.
        artifact_remover_fn: Pre-instantiated artifact remover to reuse across slides.
            See :func:`_tessellate_and_filter` for details.

    Returns:
        Dict with paths and coordinates for batch feature extraction, or ``None`` on failure.
    """
    result = _tessellate_and_filter(
        slide_path=slide_path,
        slide_id=slide_id,
        cfg=cfg,
        temp_dir=temp_dir,
        base_path=base_path,
        use_filtering=use_filtering,
        prefilter_model_type=prefilter_model_type,
        prefilter_model_path=prefilter_model_path,
        skip_second_extraction=skip_second_extraction,
        output_mask_path=output_mask_path,
        artifact_remover_fn=artifact_remover_fn,
    )
    if result is None:
        return None

    return {
        "slide_path": slide_path,
        "tessellate_h5_path": result["tessellate_h5_path"],
        "final_coords_h5_path": result["final_coords_h5_path"],
        "final_coords": result["final_coords"],
        "skip_second_extraction": skip_second_extraction,
        "prefilter_features_h5_path": result["prefilter_features_h5_path"],
    }


def _build_grid_polygons(coords, tessellate_h5_path: str) -> list:
    """Build a list of Shapely Polygons for each tile coordinate.

    Args:
        coords: Array of (x, y) tile coordinates.
        tessellate_h5_path: HDF5 file whose ``coords`` dataset carries the
            ``patch_size`` attribute used to size each polygon.

    Returns:
        List of Shapely :class:`~shapely.geometry.Polygon` objects.
    """
    with h5py.File(tessellate_h5_path, "r") as h5:
        native_patch_size = h5["coords"].attrs["patch_size"]
    polygons = []
    for coord in coords:
        x, y = coord
        polygons.append(
            Polygon(
                [
                    [x, y],
                    [x, y + native_patch_size],
                    [x + native_patch_size, y + native_patch_size],
                    [x + native_patch_size, y],
                ]
            )
        )
    return polygons


def create_visualizations(
    slide_path: str,
    final_coords,
    tessellate_h5_path: str,
    cfg,
    output_grid_mask_path: Optional[str] = None,
    output_png_dir: Optional[str] = None,
    output_thumbnail_path: Optional[str] = None,
):
    """Create optional visualizations (grid mask, PNG patches, thumbnail)."""
    # Create grid visualization
    if output_grid_mask_path:
        logger.info(f"Creating grid mask with {len(final_coords)} tiles")
        grid_polygons = _build_grid_polygons(final_coords, tessellate_h5_path)
        grid_mask = draw_slide_mask(
            slide_path,
            grid_polygons,
            **OmegaConf.to_container(cfg.vis_config),
        )
        grid_mask.save(output_grid_mask_path)

    # Save PNG patches
    if output_png_dir:
        logger.info(f"Saving patches to {output_png_dir}")
        save_patches_png(
            slide_path,
            final_coords,
            save_dir=output_png_dir,
            num_workers=cfg.num_workers,
            mpp=cfg.seg_config.mpp,
            patch_size=cfg.seg_config.patch_size,
            filter_black_white=cfg.png_config.filter_black_white,
            white_threshold=cfg.png_config.white_threshold,
            black_threshold=cfg.png_config.black_threshold,
            slide_mpp_override=cfg.seg_config.slide_mpp_override,
        )

    # Save thumbnail
    if output_thumbnail_path:
        with tiffslide.TiffSlide(slide_path) as wsi:
            logger.info(f"Saving thumbnail to {output_thumbnail_path}")
            thumbnail = wsi.get_thumbnail(cfg.thumbnail_size)
            with open(output_thumbnail_path, "wb") as f:
                thumbnail.save(f)
