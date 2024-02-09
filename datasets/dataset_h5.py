from __future__ import print_function, division

import h5py
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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


class Whole_Slide_Bag_FP(Dataset):
    def __init__(
        self,
        file_path,
        wsi,
        use_imagenet_rgb_dist=True,
        preprocess=None,
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
            self.length = len(f["coords"])

        if self.preprocess is not None:
            assert (
                use_imagenet_rgb_dist == False
            ), "Cannot use custom preprocess with ImageNet RGB dist"
            self.roi_transforms = self.preprocess
        else:
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
