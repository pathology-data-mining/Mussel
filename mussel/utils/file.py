import pickle
from contextlib import ExitStack
from pathlib import Path

import h5py


def save_pkl(filename, save_object):
    writer = open(filename, "wb")
    pickle.dump(save_object, writer)
    writer.close()


def load_pkl(filename):
    loader = open(filename, "rb")
    file = pickle.load(loader)
    loader.close()
    return file


def save_hdf5(output_path, asset_dict, attr_dict=None, attr_h5_path=None, mode="a"):
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
