import logging

import h5py

# Circular-import guard: mussel.utils.wsi_backend → mussel.utils.__init__ → feature_extract
# → mussel.datasets, which is still being initialized at this point.  Use a lazy import.
class _BackendShim:
    """Minimal shim that routes open_slide() through mussel.utils.wsi_backend."""
    @staticmethod
    def open_slide(path):
        from mussel.utils.wsi_backend import open_slide as _open_slide  # noqa: PLC0415
        return _open_slide(path)

openslide = _BackendShim()
from PIL import Image

logger = logging.getLogger(__name__)
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import eval_transforms




class WholeSlideImageH5Dataset(Dataset):
    def __init__(
        self,
        h5_path,
        slide_path,
        use_imagenet_rgb_dist=True,
        preprocess=None,
        limit_to_indices=None,
        init_wsi_in_worker=True,
    ):
        """
        Args:
                h5_path (string): Path to the .h5 file containing patched data.
                pretrained (bool): Use ImageNet transforms
                target_patch_size (int): Custom defined image size before embedding
        """
        self.use_imagenet_rgb_dist = use_imagenet_rgb_dist
        self.slide_path = slide_path
        self.wsi = None
        if not init_wsi_in_worker:
            self.wsi = openslide.open_slide(self.slide_path)
        self.limit_to_indices = limit_to_indices
        self.h5_path = h5_path

        with h5py.File(self.h5_path, "r") as f:
            self.patch_size = f["coords"].attrs["patch_size"]
            self.patch_level = f["coords"].attrs["patch_level"]
            self.scaled_patch_size = int(
                f["coords"].attrs["patch_size_to_resize_to_for_desired_mpp"]
            )
            self.length = (
                len(limit_to_indices) if limit_to_indices else len(f["coords"])
            )

        if preprocess is not None:
            assert (
                use_imagenet_rgb_dist == False
            ), "Cannot use custom preprocess with ImageNet RGB dist"
            self.roi_transforms = preprocess
        else:
            self.roi_transforms = eval_transforms(
                use_imagenet_rgb_dist=use_imagenet_rgb_dist
            )
            self.roi_transforms.transforms.insert(
                0, transforms.Resize(self.scaled_patch_size)
            )

        self.summary()

    def __len__(self):
        """Return the number of patches in the dataset."""
        return self.length

    def summary(self):
        """Print a summary of the dataset attributes and settings."""
        hdf5_file = h5py.File(self.h5_path, "r")
        dset = hdf5_file["coords"]
        for name, value in dset.attrs.items():
            logger.info(f"{name} {value}")

        hdf5_file.close()

        logger.info("\nfeature extraction settings")
        logger.info("target patch size: " + str(self.scaled_patch_size))
        logger.info("use_imagenet_rgb_dist: " + str(self.use_imagenet_rgb_dist))
        logger.info("transformations: " + str(self.roi_transforms))

    def __getitem__(self, idx_):
        """Get a patch and its coordinates by index.
        
        Args:
            idx_: Index of the patch to retrieve.
            
        Returns:
            Tuple of (transformed image tensor, coordinates).
        """
        if self.limit_to_indices:
            idx = self.limit_to_indices[idx_]
        else:
            idx = idx_

        with h5py.File(self.h5_path, "r") as hdf5_file:
            coord = hdf5_file["coords"][idx]
            try:
                img = self.wsi.read_region(
                    coord, self.patch_level, (self.patch_size, self.patch_size)
                ).convert("RGB")
                img = self.roi_transforms(img).unsqueeze(0)
                return img, coord
            except Exception as e:
                # Handle JPEG decoding errors (e.g., unsupported JPEG markers in NDPI files)
                # Return None to indicate this tile should be skipped
                if "Jpeg8Error" in str(type(e).__name__) or "imagecodecs" in str(e):
                    logger.warning(f"Skipping corrupted tile at {coord} due to JPEG decode error: {e}")
                else:
                    logger.error(f"Error reading tile at {coord}: {e}")
                return None, coord

    def worker_init(self, *args):
        """
        Needed to move wsi object creation to worker init method due to advice
        from this TiffSlide github issue:
        https://github.com/Bayer-Group/tiffslide/issues/57
        """
        self.wsi = openslide.open_slide(self.slide_path)
