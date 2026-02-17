import pickle
import tempfile
from pathlib import Path

import h5py
import numpy as np

from mussel.utils.file import load_pkl, save_hdf5, save_pkl


def test_save_and_load_pkl(tmp_path):
    """Test saving and loading pickle files"""
    test_data = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": "dict"}}
    pkl_file = tmp_path / "test.pkl"
    
    save_pkl(pkl_file, test_data)
    assert pkl_file.exists()
    
    loaded_data = load_pkl(pkl_file)
    assert loaded_data == test_data


def test_save_hdf5_creates_file(tmp_path):
    """Test that save_hdf5 creates a new HDF5 file"""
    h5_file = tmp_path / "test.h5"
    
    data = np.array([[1, 2, 3], [4, 5, 6]])
    asset_dict = {"test_data": data}
    
    save_hdf5(h5_file, asset_dict, mode="w")
    
    assert h5_file.exists()
    
    with h5py.File(h5_file, "r") as f:
        assert "test_data" in f.keys()
        np.testing.assert_array_equal(f["test_data"][:], data)


def test_save_hdf5_with_attributes(tmp_path):
    """Test saving HDF5 with attributes"""
    h5_file = tmp_path / "test_attrs.h5"
    
    data = np.array([1.0, 2.0, 3.0])
    asset_dict = {"data": data}
    attr_dict = {"data": {"description": "test data", "units": "meters"}}
    
    save_hdf5(h5_file, asset_dict, attr_dict=attr_dict, mode="w")
    
    with h5py.File(h5_file, "r") as f:
        assert f["data"].attrs["description"] == "test data"
        assert f["data"].attrs["units"] == "meters"


def test_save_hdf5_append_mode(tmp_path):
    """Test appending data to existing HDF5 file"""
    h5_file = tmp_path / "test_append.h5"
    
    # Create initial file
    data1 = np.array([[1, 2], [3, 4]])
    save_hdf5(h5_file, {"data": data1}, mode="w")
    
    # Append more data
    data2 = np.array([[5, 6], [7, 8]])
    save_hdf5(h5_file, {"data": data2}, mode="a")
    
    # Verify both datasets are present
    with h5py.File(h5_file, "r") as f:
        result = f["data"][:]
        expected = np.vstack([data1, data2])
        np.testing.assert_array_equal(result, expected)


def test_save_hdf5_with_attr_h5_path(tmp_path):
    """Test copying attributes from another HDF5 file"""
    attr_file = tmp_path / "attrs.h5"
    target_file = tmp_path / "target.h5"
    
    # Create attribute source file
    data = np.array([1, 2, 3])
    with h5py.File(attr_file, "w") as f:
        dset = f.create_dataset("data", data=data)
        dset.attrs["source"] = "test"
        dset.attrs["version"] = 1
    
    # Create target file with attributes from source
    new_data = np.array([4, 5, 6])
    save_hdf5(target_file, {"data": new_data}, attr_h5_path=attr_file, mode="w")
    
    # Verify attributes were copied
    with h5py.File(target_file, "r") as f:
        assert f["data"].attrs["source"] == "test"
        assert f["data"].attrs["version"] == 1
