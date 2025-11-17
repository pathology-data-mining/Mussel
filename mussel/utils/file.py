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


def save_hdf5(output_path, asset_dict, attr_dict=None, attr_h5_path=None, mode="a"):
    """Save data to an HDF5 file with optional attributes.
    
    Supports both local and remote paths (az://, s3://, etc.).
    For remote paths, the file is written locally first and then uploaded.
    
    Args:
        output_path: Path to the output HDF5 file (local or remote).
        asset_dict: Dictionary mapping dataset names to numpy arrays.
        attr_dict: Optional dictionary mapping dataset names to attribute dictionaries.
        attr_h5_path: Optional path to an HDF5 file to copy attributes from.
        mode: File mode ('a' for append, 'w' for write).
        
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
                    chunk_shape = (1,) + data_shape[1:]
                    maxshape = (None,) + data_shape[1:]
                    dset = file.create_dataset(
                        key,
                        shape=data_shape,
                        maxshape=maxshape,
                        chunks=chunk_shape,
                        dtype=data_type,
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
                    dset = file[key]
                    dset.resize(len(dset) + data_shape[0], axis=0)
                    dset[-data_shape[0] :] = val
        
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
