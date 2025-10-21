import pickle
from contextlib import ExitStack
from pathlib import Path

import h5py


def save_pkl(filename, save_object):
    """Save a Python object to a pickle file.
    
    Args:
        filename: Path to the output pickle file.
        save_object: Python object to serialize and save.
    """
    with open(filename, "wb") as writer:
        pickle.dump(save_object, writer)


def load_pkl(filename):
    """Load a Python object from a pickle file.
    
    Args:
        filename: Path to the pickle file to load.
        
    Returns:
        The deserialized Python object.
    """
    with open(filename, "rb") as loader:
        file = pickle.load(loader)
    return file


def save_hdf5(output_path, asset_dict, attr_dict=None, attr_h5_path=None, mode="a"):
    """Save data to an HDF5 file with optional attributes.
    
    Args:
        output_path: Path to the output HDF5 file.
        asset_dict: Dictionary mapping dataset names to numpy arrays.
        attr_dict: Optional dictionary mapping dataset names to attribute dictionaries.
        attr_h5_path: Optional path to an HDF5 file to copy attributes from.
        mode: File mode ('a' for append, 'w' for write).
        
    Returns:
        The output path.
    """
    if "w" in mode:
        Path(output_path).unlink(missing_ok=True)
    with ExitStack() as stack:
        file = stack.enter_context(h5py.File(output_path, mode))
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
    return output_path
