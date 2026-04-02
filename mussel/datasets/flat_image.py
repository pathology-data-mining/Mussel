"""
Flat image dataset for loading images from a directory without class subdirectories.
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FlatImageDataset(Dataset):
    """
    Dataset for loading images from a flat directory without class subdirectories.

    Compatible with ImageFolder interface but doesn't require class folders.
    Useful for loading patch directories where all patches are in a single folder.

    Directory structure:
        root/
            image1.png
            image2.png
            image3.png

    Instead of ImageFolder's expected structure:
        root/
            class1/
                image1.png
                image2.png
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        ),
    ):
        """
        Args:
            root: Root directory containing images
            transform: Optional transform to apply to images
            extensions: Tuple of valid image file extensions
        """
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions

        # Find all image files in the directory
        self.samples = []
        for ext in extensions:
            self.samples.extend(sorted(self.root.glob(f"*{ext}")))
            self.samples.extend(sorted(self.root.glob(f"*{ext.upper()}")))

        # Remove duplicates (in case of case-insensitive filesystem)
        self.samples = sorted(list(set(self.samples)))

        if not self.samples:
            raise FileNotFoundError(
                f"No images with extensions {extensions} found in {root}"
            )

        logger.info(f"Found {len(self.samples)} images in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Return image and dummy target (0) for compatibility with ImageFolder interface
        return img, 0
