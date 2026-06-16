import logging
import os
import pickle
import tempfile
import warnings
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import fsspec

    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient

    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False

try:
    from azure.storage.fileshare import ShareServiceClient

    AZURE_FILES_SDK_AVAILABLE = True
except ImportError:
    AZURE_FILES_SDK_AVAILABLE = False


WSI_EXTENSIONS = frozenset(
    {
        ".svs",
        ".ndpi",
        ".scn",
        ".tif",
        ".tiff",
        ".mrxs",
        ".vms",
        ".vmu",
        ".bif",
        ".qptiff",
        ".czi",
    }
)


def collect_wsi_paths(
    wsi_dir: str,
    search_nested: bool = False,
    extensions: Optional[frozenset] = None,
) -> List[str]:
    """Collect WSI file paths from a directory.

    Args:
        wsi_dir: Directory to search for WSI files.
        search_nested: If True, recursively search subdirectories.
        extensions: Set of file extensions to include (default: WSI_EXTENSIONS).

    Returns:
        Sorted list of absolute paths to WSI files found.
    """
    if extensions is None:
        extensions = WSI_EXTENSIONS
    wsi_dir_path = Path(wsi_dir)
    if not wsi_dir_path.is_dir():
        raise ValueError(f"wsi_dir is not a directory: {wsi_dir}")
    if search_nested:
        paths = [
            str(p.resolve())
            for p in wsi_dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]
    else:
        paths = [
            str(p.resolve())
            for p in wsi_dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]
    return sorted(paths)


def _is_remote_path(path):
    """Check if a path is a remote path (starts with az://, azblob://, s3://, etc.)."""
    if not isinstance(path, str):
        return False
    return path.startswith(
        ("az://", "azblob://", "abfs://", "s3://", "gs://", "http://", "https://")
    )


def is_remote_path(path):
    """Check if a path is a remote path (starts with az://, azblob://, s3://, etc.).

    Public API for checking remote paths.

    Args:
        path: Path to check (string or Path-like object)

    Returns:
        True if path is a remote URL, False otherwise
    """
    return _is_remote_path(path)


def safe_path_join(base_path, *parts):
    """Safely join path components, preserving URL schemes for remote paths.

    For remote paths (az://, s3://, etc.), uses string concatenation with /.
    For local paths, uses pathlib.Path for proper OS-specific joining.

    Args:
        base_path: Base path (can be local or remote URL)
        *parts: Path components to join

    Returns:
        Joined path as string

    Examples:
        >>> safe_path_join("s3://bucket", "folder", "file.txt")
        's3://bucket/folder/file.txt'
        >>> safe_path_join("/local/path", "folder", "file.txt")
        '/local/path/folder/file.txt'
    """
    if _is_remote_path(str(base_path)):
        # For remote paths, use string concatenation with /
        result = str(base_path).rstrip("/")
        for part in parts:
            result = f"{result}/{str(part).lstrip('/')}"
        return result
    else:
        # For local paths, use Path
        return str(Path(base_path) / Path(*parts))


def get_slide_id_from_path(slide_path: str, slide_id: Optional[str] = None) -> str:
    """Get slide ID from path or use provided ID.

    If slide_id is provided, returns it. Otherwise, extracts the filename
    without extension from slide_path.

    Args:
        slide_path: Path to the slide file
        slide_id: Optional explicit slide ID

    Returns:
        Slide ID string

    Examples:
        >>> get_slide_id_from_path("/path/to/slide.svs")
        'slide'
        >>> get_slide_id_from_path("/path/to/slide.svs", "custom_id")
        'custom_id'
    """
    return slide_id if slide_id else Path(slide_path).stem


def get_slide_ids_from_paths(
    slide_paths: List[str], slide_ids: Optional[List[str]] = None
) -> List[str]:
    """Get slide IDs from paths or use provided IDs.

    If slide_ids is provided, returns it. Otherwise, extracts filenames
    without extensions from slide_paths.

    Args:
        slide_paths: List of paths to slide files
        slide_ids: Optional list of explicit slide IDs

    Returns:
        List of slide ID strings

    Examples:
        >>> get_slide_ids_from_paths(["/a/slide1.svs", "/b/slide2.svs"])
        ['slide1', 'slide2']
        >>> get_slide_ids_from_paths(["/a/s1.svs"], ["custom"])
        ['custom']
    """
    return slide_ids if slide_ids else [Path(sp).stem for sp in slide_paths]


def ensure_directory_exists(path, is_file_path: bool = False):
    """Ensure directory exists, creating parents if needed.

    Args:
        path: Directory path or file path (str or Path)
        is_file_path: If True, creates parent directory of file

    Returns:
        Path object of the directory that was created/verified

    Examples:
        >>> ensure_directory_exists("/path/to/output")
        PosixPath('/path/to/output')
        >>> ensure_directory_exists("/path/to/file.txt", is_file_path=True)
        PosixPath('/path/to')
    """
    dir_path = Path(path).parent if is_file_path else Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def _get_fsspec_filesystem(path, ssl_verify=True):
    """Get an fsspec filesystem instance for a remote path."""
    if not FSSPEC_AVAILABLE:
        raise ImportError(
            "fsspec is required for remote file operations. Install with: pip install fsspec"
        )

    # Normalize azblob:// to az:// for fsspec
    protocol = path.split("://")[0]
    extracted_account_name = None

    if protocol == "azblob":
        protocol = "az"
        # Handle two formats:
        # 1. azblob://container/path (simple)
        # 2. azblob://account.blob.core.windows.net/container/path (full)
        remainder = path.split("://", 1)[1]
        parts = remainder.split("/", 1)

        if parts and "." in parts[0]:
            # Full format with account name
            account_part = parts[0]
            # Extract account name (first part before .blob.core.windows.net)
            if ".blob.core.windows.net" in account_part:
                extracted_account_name = account_part.split(".")[0]
            # Convert to az://container/path format
            if len(parts) > 1:
                path = f"az://{parts[1]}"
            else:
                path = f"az://"
        else:
            # Simple format: azblob://container/path
            path = "az://" + remainder

    # Get Azure credentials from environment if available
    storage_options = {}
    if path.startswith(("az://", "azblob://", "abfs://")):
        # Try to get credentials from environment
        account_name = extracted_account_name or os.environ.get(
            "AZURE_STORAGE_ACCOUNT_NAME"
        )
        account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
        sas_token = os.environ.get("AZURE_STORAGE_SAS_TOKEN")
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

        if connection_string:
            storage_options["connection_string"] = connection_string
        elif account_name and account_key:
            storage_options["account_name"] = account_name
            storage_options["account_key"] = account_key
        elif account_name and sas_token:
            storage_options["account_name"] = account_name
            storage_options["sas_token"] = sas_token
        # If no credentials, fsspec will try default Azure credentials

        # Apply SSL verification setting
        storage_options["connection_verify"] = ssl_verify
        storage_options["connection_timeout"] = 600

    return fsspec.filesystem(protocol, **storage_options)


def save_pkl(filename, save_object, ssl_verify=True):
    """Save a Python object to a pickle file.

    Supports both local and remote paths (az://, s3://, etc.).

    Args:
        filename: Path to the output pickle file (local or remote).
        save_object: Python object to serialize and save.
        ssl_verify: Whether to verify SSL certificates for remote operations.
    """
    if _is_remote_path(filename):
        fs = _get_fsspec_filesystem(filename, ssl_verify)
        with fs.open(filename, "wb") as writer:
            pickle.dump(save_object, writer)
    else:
        with open(filename, "wb") as writer:
            pickle.dump(save_object, writer)


def load_pkl(filename, ssl_verify=True):
    """Load a Python object from a pickle file.

    Supports both local and remote paths (az://, s3://, etc.).

    Args:
        filename: Path to the pickle file to load (local or remote).
        ssl_verify: Whether to verify SSL certificates for remote operations.

    Returns:
        The deserialized Python object.
    """
    if _is_remote_path(filename):
        fs = _get_fsspec_filesystem(filename, ssl_verify)
        with fs.open(filename, "rb") as loader:
            file = pickle.load(loader)
    else:
        with open(filename, "rb") as loader:
            file = pickle.load(loader)
    return file


def load_classifier(pkl_path: str):
    """Load a sklearn-style classifier from a pickle file.

    Args:
        pkl_path: Path to the classifier pickle file.

    Returns:
        The deserialized classifier object.
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_features_from_h5(h5_path: str, pt_path: Optional[str] = None):
    """Load features and coordinates from an HDF5 file.

    If *pt_path* is given and exists, features are loaded from that PyTorch
    tensor file instead (coordinates are always read from the HDF5 file).

    Args:
        h5_path: Path to the HDF5 file containing ``features`` and ``coords`` datasets.
        pt_path: Optional path to a ``.pt`` file whose tensor replaces the HDF5 features.

    Returns:
        Tuple of ``(features, coords)`` where features is a ``torch.Tensor``
        and coords is a ``numpy.ndarray``.
    """
    with h5py.File(h5_path, "r") as h5:
        if pt_path and os.path.exists(pt_path):
            features = torch.load(pt_path, weights_only=True)
        else:
            features_arr = np.array(h5["features"])
            # HDF5 stores bfloat16 as |V2 opaque void (via ml_dtypes). Cast both
            # bfloat16 and float16 to float32 so downstream consumers work correctly.
            if features_arr.dtype.kind == "V" and features_arr.dtype.itemsize == 2:
                try:
                    import ml_dtypes
                except ImportError:
                    raise ImportError(
                        "ml_dtypes is required to load bfloat16 features. "
                        "Install it with: pip install ml-dtypes"
                    )
                features_arr = features_arr.view(ml_dtypes.bfloat16).astype(np.float32)
            elif features_arr.dtype == np.float16:
                features_arr = features_arr.astype(np.float32)
            features = torch.from_numpy(features_arr)
        coords = h5["coords"][:]
    return features, coords


def save_hdf5(
    output_path,
    asset_dict,
    attr_dict=None,
    attr_h5_path=None,
    mode="a",
    compression=True,
    ssl_verify=True,
):
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
        ssl_verify: Whether to verify SSL certificates for remote operations.

    Returns:
        The output path.
    """
    is_remote = _is_remote_path(output_path)

    # For remote paths, use a temporary local file
    local_path = None  # Initialize to ensure it exists for finally block
    if is_remote:
        local_path = tempfile.NamedTemporaryFile(delete=False, suffix=".h5").name
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
                        compression_opts["compression"] = "gzip"
                        compression_opts["compression_opts"] = 4
                        # Shuffle filter can improve compression for numerical data
                        compression_opts["shuffle"] = True

                    dset = file.create_dataset(
                        key,
                        shape=data_shape,
                        maxshape=maxshape,
                        chunks=chunk_shape,
                        dtype=data_type,
                        **compression_opts,
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
            fs = _get_fsspec_filesystem(output_path, ssl_verify)
            fs.put(local_path, output_path)
    finally:
        # Clean up temporary file if used
        if is_remote and local_path is not None:
            Path(local_path).unlink(missing_ok=True)

    return output_path


def save_torch_tensor(output_path, tensor, ssl_verify=True):
    """Save a PyTorch tensor to a file.

    Supports both local and remote paths (az://, s3://, etc.).
    For remote paths, the file is written locally first and then uploaded.

    Args:
        output_path: Path to the output file (local or remote).
        tensor: PyTorch tensor to save.
        ssl_verify: Whether to verify SSL certificates for remote operations.

    Returns:
        The output path.
    """
    is_remote = _is_remote_path(output_path)

    local_path = None  # Initialize to ensure it exists for finally block
    if is_remote:
        local_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name
        try:
            torch.save(tensor, local_path)
            fs = _get_fsspec_filesystem(output_path, ssl_verify)
            fs.put(local_path, output_path)
        finally:
            if local_path is not None:
                Path(local_path).unlink(missing_ok=True)
    else:
        # Create parent directories if they don't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, output_path)

    return output_path


def download_slide(slide_path, local_dir=None, ssl_verify=True):
    """Download a remote slide to a local path.

    If the slide is already local, returns the original path.
    If the slide is remote (s3://, az://, etc.), downloads it to a local temporary location.

    Args:
        slide_path: Path to the slide (local or remote).
        local_dir: Optional directory to download to. If None, uses system temp directory.
        ssl_verify: Whether to verify SSL certificates for remote operations.

    Returns:
        Tuple of (local_path, is_temp) where is_temp indicates if cleanup is needed.
    """
    if not _is_remote_path(slide_path):
        # Already local
        return slide_path, False

    # Download remote file
    if local_dir is None:
        local_dir = tempfile.mkdtemp(prefix="mussel_slides_")
    else:
        os.makedirs(local_dir, exist_ok=True)

    # Extract filename from remote path
    filename = Path(slide_path).name
    local_path = os.path.join(local_dir, filename)

    logger.info(f"Downloading remote slide {slide_path} to {local_path}")
    try:
        fs = _get_fsspec_filesystem(slide_path, ssl_verify)
        fs.get(slide_path, local_path)
        logger.info(f"Download complete: {local_path}")
        return local_path, True
    except Exception as e:
        logger.error(f"Failed to download {slide_path}: {e}")
        raise


def _download_azure_directory_with_sdk(container_name, prefix, local_path):
    """Download an Azure blob directory using the Azure SDK directly (faster than fsspec).

    Args:
        container_name: Azure blob container name
        prefix: Blob prefix (directory path)
        local_path: Local destination directory
    """
    if not AZURE_SDK_AVAILABLE:
        raise ImportError("azure-storage-blob is required for Azure downloads")

    # Suppress Azure SDK and urllib3 logging to reduce noise
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=Warning, module="urllib3")

    # Get credentials
    account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")

    if not account_name or not account_key:
        raise ValueError(
            "AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY must be set"
        )

    account_url = f"https://{account_name}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(
        account_url=account_url, credential=account_key, connection_verify=False
    )

    container_client = blob_service_client.get_container_client(container_name)

    # List and download all blobs with the prefix
    logger.info(f"Listing blobs in {container_name}/{prefix}")
    blob_count = 0
    downloaded_count = 0
    skipped_count = 0

    for blob in container_client.list_blobs(name_starts_with=prefix):
        blob_count += 1

        # Remove prefix to get relative path
        relative_path = blob.name[len(prefix) :].lstrip("/")
        if not relative_path:
            # Skip directory marker blobs or empty paths
            skipped_count += 1
            continue

        local_file_path = os.path.join(local_path, relative_path)

        # Skip if file already exists (avoid re-downloading)
        if (
            os.path.exists(local_file_path)
            and os.path.getsize(local_file_path) == blob.size
        ):
            skipped_count += 1
            continue

        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Only log every 10th download to reduce noise
        if downloaded_count % 10 == 0:
            logger.info(f"Downloading {blob.name} ({blob.size} bytes)...")

        blob_client = container_client.get_blob_client(blob.name)
        with open(local_file_path, "wb") as f:
            blob_client.download_blob().readinto(f)
        downloaded_count += 1

    logger.info(
        f"Download complete: {downloaded_count} files downloaded, {skipped_count} skipped"
    )


def _download_azure_files_directory(share_name, prefix, local_path):
    """Download a directory from Azure Files using the Azure SDK.

    Args:
        share_name: Azure Files share name
        prefix: Directory prefix within the share
        local_path: Local destination directory
    """
    if not AZURE_FILES_SDK_AVAILABLE:
        raise ImportError(
            "azure-storage-file-share is required for Azure Files downloads"
        )

    # Suppress Azure SDK logging
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure.storage.fileshare").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=Warning, module="urllib3")

    # Get credentials
    account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")

    if not account_name or not account_key:
        raise ValueError(
            "AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY must be set"
        )

    account_url = f"https://{account_name}.file.core.windows.net"
    share_service_client = ShareServiceClient(
        account_url=account_url, credential=account_key
    )

    share_client = share_service_client.get_share_client(share_name)

    logger.info(f"Listing files in {share_name}/{prefix}")
    downloaded_count = 0
    skipped_count = 0

    # Recursively download directory contents
    def download_directory(dir_path, local_dir):
        nonlocal downloaded_count, skipped_count

        dir_client = share_client.get_directory_client(dir_path)

        for item in dir_client.list_directories_and_files():
            item_name = item["name"]
            remote_item_path = f"{dir_path}/{item_name}" if dir_path else item_name
            local_item_path = os.path.join(local_dir, item_name)

            if item.get("is_directory", False):
                # Recursively download subdirectory
                os.makedirs(local_item_path, exist_ok=True)
                download_directory(remote_item_path, local_item_path)
            else:
                # Download file
                file_size = item.get("content_length", 0)

                # Skip if file already exists with same size
                if (
                    os.path.exists(local_item_path)
                    and os.path.getsize(local_item_path) == file_size
                ):
                    skipped_count += 1
                    continue

                # Only log every 10th download to reduce noise
                if downloaded_count % 10 == 0:
                    logger.info(
                        f"Downloading {remote_item_path} ({file_size} bytes)..."
                    )

                file_client = share_client.get_file_client(remote_item_path)
                with open(local_item_path, "wb") as f:
                    data = file_client.download_file()
                    f.write(data.readall())
                downloaded_count += 1

    download_directory(prefix.rstrip("/"), local_path)
    logger.info(
        f"Download complete: {downloaded_count} files downloaded, {skipped_count} skipped"
    )


def _download_azure_files_file(share_name, file_path, local_path):
    """Download a single file from Azure Files using the Azure SDK.

    Args:
        share_name: Azure Files share name
        file_path: Path to the file within the share
        local_path: Local destination file path
    """
    if not AZURE_FILES_SDK_AVAILABLE:
        raise ImportError(
            "azure-storage-file-share is required for Azure Files downloads"
        )

    # Suppress Azure SDK logging
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure.storage.fileshare").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=Warning, module="urllib3")

    # Get credentials
    account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")

    if not account_name or not account_key:
        raise ValueError(
            "AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY must be set"
        )

    account_url = f"https://{account_name}.file.core.windows.net"
    share_service_client = ShareServiceClient(
        account_url=account_url, credential=account_key
    )

    share_client = share_service_client.get_share_client(share_name)
    file_client = share_client.get_file_client(file_path)

    logger.info(f"Downloading single file {share_name}/{file_path}")

    # Create parent directory if needed
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Download file
    with open(local_path, "wb") as f:
        data = file_client.download_file()
        f.write(data.readall())

    logger.info(f"Download complete: {local_path}")


def download_model_path(model_path, cache_dir=None, ssl_verify=True):
    """Download a remote model path to a local cache directory.

    If the model path is already local, returns the original path.
    If the model path is remote (s3://, az://, etc.), downloads it to a cache directory.
    Supports both single files and directories.

    Args:
        model_path: Path to the model file or directory (local or remote).
        cache_dir: Optional cache directory to download to. If None, uses HF_HOME or system default.
        ssl_verify: Whether to verify SSL certificates for remote operations.

    Returns:
        Local path to the downloaded model (file or directory).
    """
    if not _is_remote_path(model_path):
        # Already local
        return model_path

    # Determine cache directory
    if cache_dir is None:
        cache_dir = (
            os.environ.get("HF_HOME")
            or os.environ.get("TRANSFORMERS_CACHE")
            or os.path.expanduser("~/.cache/mussel")
        )

    # Create cache subdirectory for models
    models_cache_dir = os.path.join(cache_dir, "remote_models")
    os.makedirs(models_cache_dir, exist_ok=True)

    # Extract the base name from remote path to create a cache location
    # For URLs like az://container/models/GIGAPATH_SLIDE, we want to preserve the structure
    path_parts = model_path.split("://", 1)[1]  # Remove scheme
    # Replace slashes with underscores to create a unique cache key
    cache_key = path_parts.replace("/", "_").replace("\\", "_")
    local_path = os.path.join(models_cache_dir, cache_key)

    # Check if already cached
    if os.path.exists(local_path):
        logger.info(f"Using cached model from {local_path}")
        return local_path

    # Download remote model
    logger.info(f"Downloading remote model {model_path} to {local_path}")
    try:
        # For Azure Files paths (azfiles://), use Azure Files SDK
        if model_path.startswith("azfiles://") and AZURE_FILES_SDK_AVAILABLE:
            logger.info("Using Azure Files SDK for download")
            # Parse share name and path: azfiles://share/path/to/model
            path_parts = model_path.split("://", 1)[1]
            parts = path_parts.split("/", 1)

            share_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""

            # Azure Files paths ending with / are directories
            if model_path.endswith("/") or not prefix:
                # Directory download
                os.makedirs(local_path, exist_ok=True)
                _download_azure_files_directory(share_name, prefix, local_path)
            else:
                # Single file download - check if path contains subdirectories
                if "/" in prefix:
                    # Path like "models/subdir/file.pth" - treat as directory structure
                    os.makedirs(local_path, exist_ok=True)
                    _download_azure_files_directory(share_name, prefix, local_path)
                else:
                    # Path like "model.pth" - single file in share root
                    _download_azure_files_file(share_name, prefix, local_path)
        # For Azure Blob paths, use direct Azure SDK (more reliable than fsspec)
        elif (
            model_path.startswith(("az://", "azblob://", "abfs://"))
            and AZURE_SDK_AVAILABLE
        ):
            logger.info("Using Azure SDK for download (more reliable than fsspec)")
            # Parse container and prefix, handling both azblob formats:
            # 1. azblob://container/prefix/path
            # 2. azblob://account.blob.core.windows.net/container/prefix/path
            path_parts = model_path.split("://", 1)[1]
            parts = path_parts.split("/")

            # Check if first part contains dots (account name format)
            if "." in parts[0]:
                # Full format: skip account part, use next as container
                container_name = parts[1] if len(parts) > 1 else ""
                prefix = "/".join(parts[2:]) if len(parts) > 2 else ""
            else:
                # Simple format: first part is container
                container_name = parts[0]
                prefix = "/".join(parts[1:]) if len(parts) > 1 else ""

            # Azure paths ending with / are directories
            if model_path.endswith("/"):
                os.makedirs(local_path, exist_ok=True)
                _download_azure_directory_with_sdk(container_name, prefix, local_path)
            else:
                # Single file - fall back to fsspec
                logger.warning(
                    f"Single file download from Azure via SDK not implemented, using fsspec"
                )
                fs = _get_fsspec_filesystem(model_path, ssl_verify)
                fs.get(model_path, local_path)
        else:
            # Use fsspec for non-Azure or if Azure SDK unavailable
            fs = _get_fsspec_filesystem(model_path, ssl_verify)

            # For Azure blob storage, simply check if path ends with / to determine if directory
            # Avoid pre-checking with ls() or isdir() as these can hang due to network issues
            is_directory = model_path.endswith("/")

            if is_directory:
                # Download directory recursively
                logger.info(f"Downloading directory {model_path} recursively")
                fs.get(model_path.rstrip("/") + "/", local_path, recursive=True)
            else:
                # Download single file
                logger.info(f"Downloading file {model_path}")
                fs.get(model_path, local_path)

        logger.info(f"Download complete: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download model from {model_path}: {e}")
        raise


def resolve_remote_paths(*attrs, auto_detect=True, suffixes=None, ssl_verify=True):
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
        ssl_verify: Whether to verify SSL certificates for remote operations.

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
    # Default suffixes for auto-detection
    if suffixes is None:
        suffixes = ["_path", "_dir", "_pkl", "_file", "_model"]

    def decorator(func):
        @wraps(func)
        def wrapper(cfg, *args, **kwargs):
            # Collect attributes to check
            attrs_to_check = set(attrs) if attrs else set()

            # Auto-detect path-like attributes if enabled
            if auto_detect:
                for attr in dir(cfg):
                    # Skip private/magic attributes
                    if attr.startswith("_"):
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
                        local_path = download_model_path(value, ssl_verify=ssl_verify)
                        setattr(cfg, attr, local_path)
                        logger.info(f"Resolved {attr} to local path: {local_path}")
                    except Exception as e:
                        logger.error(f"Failed to download remote {attr} '{value}': {e}")
                        # Continue execution - let the original code handle missing paths
                else:
                    logger.debug(f"{attr} is already a local path: {value}")

            return func(cfg, *args, **kwargs)

        return wrapper

    return decorator
