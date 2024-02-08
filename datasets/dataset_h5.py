from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py


def eval_transforms(use_imagenet_rgb_dist=False):
    if use_imagenet_rgb_dist:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    trnsfrms_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    return trnsfrms_val


class Whole_Slide_Bag(Dataset):
    def __init__(
        self,
        file_path,
        use_imagenet_rgb_dist=True,
        target_patch_size=-1,
    ):
        """
        Args:
                file_path (string): Path to the .h5 file containing patched data.
                pretrained (bool): Use ImageNet transforms
                custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained = pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        if not custom_transforms:
            self.roi_transforms = eval_transforms(
                use_imagenet_rgb_dist=use_imagenet_rgb_dist
            )
        else:
            raise NotImplementedError  # self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f["imgs"]
            self.length = len(dset)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file["imgs"]
        for name, value in dset.attrs.items():
            print(name, value)

        print("pretrained:", self.pretrained)
        print("transformations:", self.roi_transforms)
        if self.target_patch_size is not None:
            print("target_size: ", self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as hdf5_file:
            img = hdf5_file["imgs"][idx]
            coord = hdf5_file["coords"][idx]

        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


class Whole_Slide_Bag_FP(Dataset):
    def __init__(
        self,
        file_path,
        wsi,
        use_imagenet_rgb_dist=True,
    ):
        """
        Args:
                file_path (string): Path to the .h5 file containing patched data.
                pretrained (bool): Use ImageNet transforms
                target_patch_size (int): Custom defined image size before embedding
        """
        self.use_imagenet_rgb_dist = use_imagenet_rgb_dist
        self.wsi = wsi

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            self.patch_size = f["coords"].attrs["patch_size"]
            self.patch_level = f["coords"].attrs["patch_level"]
            self.scaled_patch_size = int(
                f["coords"].attrs["patch_size_to_resize_to_for_desired_mpp"]
            )

        self.roi_transforms = eval_transforms(
            use_imagenet_rgb_dist=use_imagenet_rgb_dist
        )
        self.roi_transforms.transforms.insert(
            0, transforms.Resize(self.scaled_patch_size)
        )

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file["coords"]
        for name, value in dset.attrs.items():
            print(name, value)

        print("\nfeature extraction settings")
        print("target patch size: ", self.scaled_patch_size)
        print("use_imagenet_rgb_dist: ", self.use_imagenet_rgb_dist)
        print("transformations: ", self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as hdf5_file:
            coord = hdf5_file["coords"][idx]
        img = self.wsi.read_region(
            coord, self.patch_level, (self.patch_size, self.patch_size)
        ).convert("RGB")

        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord
