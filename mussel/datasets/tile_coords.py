import tiffslide as openslide
from loguru import logger
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import eval_transforms

class WholeSlideImageTileCoordDataset(Dataset):
    def __init__(
        self,
        coords,
        attrs,
        slide_path,
        use_imagenet_rgb_dist=True,
        preprocess=None,
        limit_to_indices=None,
    ):
        """
        Args:
                coords (list of tuples): List of (x, y) coordinates for patches.
                attrs (dict): Attributes including 'patch_size', 'patch_level', and 'scaled_patch_size'.
                slide_path (string): Path to the whole slide image file.
                use_imagenet_rgb_dist (bool): Use ImageNet RGB distribution for normalization.
                preprocess (callable, optional): Custom preprocessing function. Defaults to None.
                limit_to_indices (list of int, optional): Limit dataset to these indices. Defaults to None.
        """
        self.use_imagenet_rgb_dist = use_imagenet_rgb_dist
        self.slide_path = slide_path
        self.wsi = None
        self.limit_to_indices = limit_to_indices
        self.coords = coords
        self.patch_size = attrs["patch_size"]
        self.patch_level = attrs["patch_level"]
        self.scaled_patch_size = attrs["patch_size_to_resize_to_for_desired_mpp"]

        self.length = (
            len(limit_to_indices) if limit_to_indices else len(coords)
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
        return self.length

    def summary(self):
        logger.info("\nfeature extraction settings")
        logger.info("target patch size: " + str(self.scaled_patch_size))
        logger.info("use_imagenet_rgb_dist: " + str(self.use_imagenet_rgb_dist))
        logger.info("transformations: " + str(self.roi_transforms))

    def __getitem__(self, idx_):
        if self.limit_to_indices:
            idx = self.limit_to_indices[idx_]
        else:
            idx = idx_

        coord = self.coords[idx]
        img = self.wsi.read_region(
            coord, self.patch_level, (self.patch_size, self.patch_size)
        ).convert("RGB")

        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord

    def worker_init(self, *args):
        """
        Needed to move wsi object creation to worker init method due to advice
        from this TiffSlide github issue:
        https://github.com/Bayer-Group/tiffslide/issues/57
        """
        self.wsi = openslide.open_slide(self.slide_path)
