from .artifact_removal import GrandQCArtifactRemover
from .converter import AnyToTiffConverter
from .file import (WSI_EXTENSIONS, collect_wsi_paths, download_model_path,
                   ensure_directory_exists, get_slide_id_from_path,
                   get_slide_ids_from_paths, is_remote_path, load_classifier,
                   load_features_from_h5, resolve_remote_paths, safe_path_join,
                   save_hdf5, save_torch_tensor)
from .ml import collate_features
from .segment import (contours_to_polygon, draw_slide_mask,
                      get_level_for_magnification, get_slide_mpp,
                      save_patches_png, segment_tissue)
from .tile_export import export_tiles
from .timer import timed
from .visualization import visualize_heatmap
from .wsi_backend import open_slide

_FEATURE_EXTRACT_EXPORTS = {
    "DatasetProcessor",
    "FeatureExtractionResult",
    "H5DatasetProcessor",
    "ImageFolderProcessor",
    "TileCoordProcessor",
    "aggregate_sample_features",
    "aggregate_slide_features",
    "aggregate_slide_features_batch",
    "extract_patch_features",
    "extract_patch_features_batch",
    "filter_features",
    "get_batch_size_for_model",
    "get_classifier_pkl_from_model_dir",
    "get_dataset_processor",
    "get_features",
    "get_model_path_from_dir",
    "process_dataset",
    "resolve_aggregation_method",
    "resolve_patch_encoder",
    "save_features",
    "subsample_tiles",
}


def __getattr__(name):
    if name in _FEATURE_EXTRACT_EXPORTS:
        from . import feature_extract

        value = getattr(feature_extract, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GrandQCArtifactRemover",
    "AnyToTiffConverter",
    "WSI_EXTENSIONS",
    "collect_wsi_paths",
    "download_model_path",
    "ensure_directory_exists",
    "get_slide_id_from_path",
    "get_slide_ids_from_paths",
    "is_remote_path",
    "load_classifier",
    "load_features_from_h5",
    "resolve_remote_paths",
    "safe_path_join",
    "save_hdf5",
    "save_torch_tensor",
    "collate_features",
    "contours_to_polygon",
    "draw_slide_mask",
    "get_level_for_magnification",
    "get_slide_mpp",
    "save_patches_png",
    "segment_tissue",
    "export_tiles",
    "timed",
    "visualize_heatmap",
    "open_slide",
    *_FEATURE_EXTRACT_EXPORTS,
]
