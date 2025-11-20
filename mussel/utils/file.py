import pickle
import os
import ssl
from contextlib import ExitStack
from pathlib import Path

import h5py

# Disable SSL verification globally (like in CLI scripts)
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import fsspec
    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False


def _is_remote_path(path):
    """Check if a path is a remote path (starts with az://, s3://, etc.)."""
    if not isinstance(path, str):
        return False
    return path.startswith(('az://', 'abfs://', 's3://', 'gs://', 'http://', 'https://'))


def _get_fsspec_filesystem(path):
    """Get an fsspec filesystem instance for a remote path."""
    if not FSSPEC_AVAILABLE:
        raise ImportError("fsspec is required for remote file operations. Install with: pip install fsspec")
    
    # Get Azure credentials from environment if available
    storage_options = {}
    if path.startswith(('az://', 'abfs://')):
        # Try to get credentials from environment
        account_name = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME')
        account_key = os.environ.get('AZURE_STORAGE_ACCOUNT_KEY')
        sas_token = os.environ.get('AZURE_STORAGE_SAS_TOKEN')
        connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        
        if connection_string:
            storage_options['connection_string'] = connection_string
        elif account_name and account_key:
            storage_options['account_name'] = account_name
            storage_options['account_key'] = account_key
        elif account_name and sas_token:
            storage_options['account_name'] = account_name
            storage_options['sas_token'] = sas_token
        # If no credentials, fsspec will try default Azure credentials
        
        # For Azure, disable SSL verification - try both methods
        storage_options['connection_verify'] = False
        # Also set the SSL context to not verify certificates
        storage_options['connection_timeout'] = 600
        storage_options['connection_cert'] = None
    
    return fsspec.filesystem(path.split('://')[0], **storage_options)


def save_pkl(filename, save_object):
    """Save a Python object to a pickle file.
    
    Supports both local and remote paths (az://, s3://, etc.).
    
    Args:
        filename: Path to the output pickle file (local or remote).
        save_object: Python object to serialize and save.
    """
    if _is_remote_path(filename):
        fs = _get_fsspec_filesystem(filename)
        with fs.open(filename, "wb") as writer:
            pickle.dump(save_object, writer)
    else:
        with open(filename, "wb") as writer:
            pickle.dump(save_object, writer)


def load_pkl(filename):
    """Load a Python object from a pickle file.
    
    Supports both local and remote paths (az://, s3://, etc.).
    
    Args:
        filename: Path to the pickle file to load (local or remote).
        
    Returns:
        The deserialized Python object.
    """
    if _is_remote_path(filename):
        fs = _get_fsspec_filesystem(filename)
        with fs.open(filename, "rb") as loader:
            file = pickle.load(loader)
    else:
        with open(filename, "rb") as loader:
            file = pickle.load(loader)
    return file


def save_hdf5(output_path, asset_dict, attr_dict=None, attr_h5_path=None, mode="a", compression=True):
    """Save data to an HDF5 file with optional attributes.
    
    Supports both local and remote paths (az://, s3://, etc.).
    For remote paths, the file is written locally first and then uploaded.
    
    Optimizations applied:
    - Larger chunk sizes (min 128 rows) for better I/O performance
    - Optional gzip compression (enabled by default) to reduce file size 3-4x
    - Efficient resize operations with proper chunk alignment
    
    Args:
        output_path: Path to the output HDF5 file (local or remote).
        asset_dict: Dictionary mapping dataset names to numpy arrays.
        attr_dict: Optional dictionary mapping dataset names to attribute dictionaries.
        attr_h5_path: Optional path to an HDF5 file to copy attributes from.
        mode: File mode ('a' for append, 'w' for write).
        compression: Enable gzip compression (default: True). Reduces file size 3-4x
                    with minimal performance impact. Set to False for uncompressed.
        
    Returns:
        The output path.
    """
    is_remote = _is_remote_path(output_path)
    
    # For remote paths, use a temporary local file
    if is_remote:
        import tempfile
        local_path = tempfile.NamedTemporaryFile(delete=False, suffix='.h5').name
        actual_path = local_path
    else:
        actual_path = output_path
        # Create parent directories if they don't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if "w" in mode:
            Path(output_path).unlink(missing_ok=True)
    
    try:
        with ExitStack() as stack:
            file = stack.enter_context(h5py.File(actual_path, mode))
            if attr_h5_path is not None:
                attr_file = stack.enter_context(h5py.File(attr_h5_path))

            for key, val in asset_dict.items():
                data_shape = val.shape
                if key not in file:
                    data_type = val.dtype
                    
                    # Optimized chunking: Use larger chunk size (min 128 rows)
                    # This reduces metadata overhead and improves I/O performance
                    # For batches of 128, this means 1 chunk per batch instead of 128 chunks
                    chunk_rows = max(128, data_shape[0])  # At least 128 rows per chunk
                    chunk_shape = (chunk_rows,) + data_shape[1:]
                    maxshape = (None,) + data_shape[1:]
                    
                    # Auto-detect if this is patch tokens (3D with many tokens)
                    # ViT models return (batch, num_tokens, embed_dim) - e.g., (64, 257, 1280)
                    # Aggregated features are (batch, embed_dim) - e.g., (64, 1280)
                    is_patch_tokens = len(data_shape) == 3 and data_shape[1] > 100
                    
                    # Setup compression if enabled
                    # Disable compression for patch tokens (high-entropy ViT features don't compress well)
                    compression_opts = {}
                    use_compression = compression and not is_patch_tokens
                    
                    if use_compression:
                        # gzip level 4 is a good balance between speed and compression ratio
                        # Typically achieves 3-4x compression for aggregated embeddings
                        compression_opts['compression'] = 'gzip'
                        compression_opts['compression_opts'] = 4
                        # Shuffle filter can improve compression for numerical data
                        compression_opts['shuffle'] = True
                    
                    dset = file.create_dataset(
                        key,
                        shape=data_shape,
                        maxshape=maxshape,
                        chunks=chunk_shape,
                        dtype=data_type,
                        **compression_opts
                    )
                    dset[:] = val
                    if attr_dict is not None:
                        if key in attr_dict.keys():
                            for attr_key, attr_val in attr_dict[key].items():
                                dset.attrs[attr_key] = attr_val
                    if attr_h5_path is not None:
                        if key in attr_file.keys():
                            for attr_key, attr_val in attr_file[key].attrs.items():
                                dset.attrs[attr_key] = attr_val
                else:
                    # Append mode: resize and add new data
                    dset = file[key]
                    old_len = len(dset)
                    new_len = old_len + data_shape[0]
                    dset.resize(new_len, axis=0)
                    dset[old_len:new_len] = val
        
        # Upload to remote if needed
        if is_remote:
            fs = _get_fsspec_filesystem(output_path)
            fs.put(local_path, output_path)
    finally:
        # Clean up temporary file if used
        if is_remote:
            Path(local_path).unlink(missing_ok=True)
    
    return output_path


def save_torch_tensor(output_path, tensor):
    """Save a PyTorch tensor to a file.
    
    Supports both local and remote paths (az://, s3://, etc.).
    For remote paths, the file is written locally first and then uploaded.
    
    Args:
        output_path: Path to the output file (local or remote).
        tensor: PyTorch tensor to save.
        
    Returns:
        The output path.
    """
    import torch
    
    is_remote = _is_remote_path(output_path)
    
    if is_remote:
        import tempfile
        local_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pt').name
        try:
            torch.save(tensor, local_path)
            fs = _get_fsspec_filesystem(output_path)
            fs.put(local_path, output_path)
        finally:
            Path(local_path).unlink(missing_ok=True)
    else:
        # Create parent directories if they don't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, output_path)
    
    return output_path


def download_slide(slide_path, local_dir=None):
    """Download a remote slide to a local path.

    If the slide is already local, returns the original path.
    If the slide is remote (s3://, az://, etc.), downloads it to a local temporary location.

    Args:
        slide_path: Path to the slide (local or remote).
        local_dir: Optional directory to download to. If None, uses system temp directory.

    Returns:
        Tuple of (local_path, is_temp) where is_temp indicates if cleanup is needed.
    """
    from loguru import logger

    if not _is_remote_path(slide_path):
        # Already local
        return slide_path, False

    # Download remote file
    import tempfile
    if local_dir is None:
        local_dir = tempfile.mkdtemp(prefix='mussel_slides_')
    else:
        os.makedirs(local_dir, exist_ok=True)

    # Extract filename from remote path
    filename = Path(slide_path).name
    local_path = os.path.join(local_dir, filename)

    logger.info(f"Downloading remote slide {slide_path} to {local_path}")
    try:
        fs = _get_fsspec_filesystem(slide_path)
        fs.get(slide_path, local_path)
        logger.info(f"Download complete: {local_path}")
        return local_path, True
    except Exception as e:
        logger.error(f"Failed to download {slide_path}: {e}")
        raise


def download_model_path(model_path, cache_dir=None):
    """Download a remote model path to a local cache directory.

    If the model path is already local, returns the original path.
    If the model path is remote (s3://, az://, etc.), downloads it to a cache directory.
    Supports both single files and directories.

    Args:
        model_path: Path to the model file or directory (local or remote).
        cache_dir: Optional cache directory to download to. If None, uses HF_HOME or system default.

    Returns:
        Local path to the downloaded model (file or directory).
    """
    from loguru import logger

    if not _is_remote_path(model_path):
        # Already local
        return model_path

    # Determine cache directory
    if cache_dir is None:
        cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or os.path.expanduser('~/.cache/mussel')

    # Create cache subdirectory for models
    models_cache_dir = os.path.join(cache_dir, 'remote_models')
    os.makedirs(models_cache_dir, exist_ok=True)

    # Extract the base name from remote path to create a cache location
    # For URLs like az://container/models/GIGAPATH_SLIDE, we want to preserve the structure
    path_parts = model_path.split('://', 1)[1]  # Remove scheme
    # Replace slashes with underscores to create a unique cache key
    cache_key = path_parts.replace('/', '_').replace('\\', '_')
    local_path = os.path.join(models_cache_dir, cache_key)

    # Check if already cached
    if os.path.exists(local_path):
        logger.info(f"Using cached model from {local_path}")
        return local_path

    # Download remote model
    logger.info(f"Downloading remote model {model_path} to {local_path}")
    try:
        fs = _get_fsspec_filesystem(model_path)

        # Check if remote path is a directory or file
        try:
            # Try to list contents to see if it's a directory
            is_directory = fs.isdir(model_path)
        except Exception:
            # If we can't determine, assume it's a file
            is_directory = False

        if is_directory:
            # Download directory recursively
            logger.info(f"Downloading directory {model_path}")
            fs.get(model_path, local_path, recursive=True)
        else:
            # Download single file
            logger.info(f"Downloading file {model_path}")
            fs.get(model_path, local_path)

        logger.info(f"Download complete: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download model from {model_path}: {e}")
        raise


def resolve_remote_paths(*attrs, auto_detect=True, suffixes=None):
    """
    Decorator that automatically resolves remote paths in config to local cached paths.

    Checks specified config attributes (or auto-detects them) for remote paths
    (az://, s3://, gs://, etc.) and downloads them to a local cache directory
    before the function executes.

    Args:
        *attrs: Specific attribute names to check and resolve. If provided along with
                auto_detect=True, these will be checked in addition to auto-detected attrs.
        auto_detect: If True, automatically detect attributes ending with common path suffixes.
                    Default: True
        suffixes: List of suffixes to detect (e.g., ['_path', '_dir', '_pkl']).
                 If None, uses default: ['_path', '_dir', '_pkl', '_file']

    Usage:
        # Auto-detect all path-like attributes
        @resolve_remote_paths()
        def process_slides(cfg):
            ...

        # Specify exact attributes
        @resolve_remote_paths('model_path', 'classifier_pkl', auto_detect=False)
        def process_slides(cfg):
            ...

        # Combine explicit + auto-detection
        @resolve_remote_paths('custom_attr', auto_detect=True)
        def process_slides(cfg):
            ...
    """
    from functools import wraps
    from loguru import logger

    # Default suffixes for auto-detection
    if suffixes is None:
        suffixes = ['_path', '_dir', '_pkl', '_file', '_model']

    def decorator(func):
        @wraps(func)
        def wrapper(cfg, *args, **kwargs):
            # Collect attributes to check
            attrs_to_check = set(attrs) if attrs else set()

            # Auto-detect path-like attributes if enabled
            if auto_detect:
                for attr in dir(cfg):
                    # Skip private/magic attributes
                    if attr.startswith('_'):
                        continue
                    # Check if attribute ends with common path suffixes
                    if any(attr.endswith(suffix) for suffix in suffixes):
                        attrs_to_check.add(attr)

            # Resolve each attribute
            for attr in sorted(attrs_to_check):
                if not hasattr(cfg, attr):
                    continue

                value = getattr(cfg, attr)

                # Skip None values and non-string values
                if value is None or not isinstance(value, str):
                    continue

                # Check if it's a remote path
                if _is_remote_path(value):
                    logger.info(f"Downloading remote {attr}: {value}")
                    try:
                        local_path = download_model_path(value)
                        setattr(cfg, attr, local_path)
                        logger.info(f"Resolved {attr} to local path: {local_path}")
                    except Exception as e:
                        logger.error(f"Failed to download remote {attr} '{value}': {e}")
                        # Continue execution - let the original code handle missing paths

            return func(cfg, *args, **kwargs)

        return wrapper

    return decorator
