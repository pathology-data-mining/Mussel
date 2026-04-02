from torchvision import transforms

from mussel.models.base import IMAGENET_MEAN, IMAGENET_STD


def eval_transforms(use_imagenet_rgb_dist=False):
    """Create transforms for evaluation/inference.

    Args:
        use_imagenet_rgb_dist: Use ImageNet RGB distribution for normalization (default: False).

    Returns:
        Composed torchvision transforms for image preprocessing.
    """
    if use_imagenet_rgb_dist:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    transforms_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    return transforms_val
