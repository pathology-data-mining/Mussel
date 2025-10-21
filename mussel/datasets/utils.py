from torchvision import transforms

def eval_transforms(use_imagenet_rgb_dist=False):
    """Create transforms for evaluation/inference.
    
    Args:
        use_imagenet_rgb_dist: Use ImageNet RGB distribution for normalization (default: False).
        
    Returns:
        Composed torchvision transforms for image preprocessing.
    """
    if use_imagenet_rgb_dist:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    transforms_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    return transforms_val
