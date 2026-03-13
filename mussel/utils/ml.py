import collections
import logging
import math
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (DataLoader, RandomSampler, Sampler,
                              SequentialSampler, WeightedRandomSampler,
                              sampler)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
            indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        """Initialize the sampler with a list of indices.
        
        Args:
            indices: A sequence of indices to sample from.
        """
        self.indices = indices

    def __iter__(self):
        """Return an iterator over the indices."""
        return iter(self.indices)

    def __len__(self):
        """Return the number of indices."""
        return len(self.indices)


def collate_MIL(batch):
    """Collate function for Multiple Instance Learning (MIL) batches.
    
    Args:
        batch: List of tuples containing (image, label) pairs.
        
    Returns:
        List containing [concatenated images, labels tensor].
    """
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def collate_features(batch):
    """Collate function for feature batches with coordinates.
    
    Args:
        batch: List of tuples containing (features, coordinates) pairs.
        
    Returns:
        List containing [concatenated features, stacked coordinates].
    """
    # Filter out None values (corrupted/failed tiles)
    batch = [item for item in batch if item[0] is not None]
    
    # If all tiles in batch failed, return empty tensors
    if len(batch) == 0:
        return [torch.empty(0), np.empty((0, 2))]
    
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
    """Create a simple DataLoader with sequential sampling.
    
    Args:
        dataset: Dataset to load.
        batch_size: Number of samples per batch (default: 1).
        num_workers: Number of worker processes for data loading (default: 1).
        
    Returns:
        DataLoader instance.
    """
    kwargs = (
        {"num_workers": num_workers, "pin_memory": False}
        if device.type == "cuda"
        else {}
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler.SequentialSampler(dataset),
        collate_fn=collate_MIL,
        **kwargs,
    )
    return loader


def get_split_loader(split_dataset, training=False, testing=False, weighted=False):
    """
    Return either the validation loader or training loader
    """
    kwargs = {"num_workers": 4} if device.type == "cuda" else {}
    
    if testing:
        ids = np.random.choice(
            np.arange(len(split_dataset), int(len(split_dataset) * 0.1)), replace=False
        )
        loader = DataLoader(
            split_dataset,
            batch_size=1,
            sampler=SubsetSequentialSampler(ids),
            collate_fn=collate_MIL,
            **kwargs,
        )
        return loader
    
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            sampler_instance = WeightedRandomSampler(weights, len(weights))
        else:
            sampler_instance = RandomSampler(split_dataset)
    else:
        sampler_instance = SequentialSampler(split_dataset)
    
    loader = DataLoader(
        split_dataset,
        batch_size=1,
        sampler=sampler_instance,
        collate_fn=collate_MIL,
        **kwargs,
    )
    
    return loader


def get_optim(model, args):
    """Create an optimizer for the model.
    
    Args:
        model: The neural network model.
        args: Arguments object containing opt (optimizer type), lr (learning rate), and reg (weight decay).
        
    Returns:
        Optimizer instance (Adam or SGD).
        
    Raises:
        NotImplementedError: If optimizer type is not supported.
    """
    if args.opt == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.reg,
        )
    elif args.opt == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.reg,
        )
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    """Print network architecture and parameter counts.
    
    Args:
        net: Neural network model to analyze.
    """
    num_params = 0
    num_params_train = 0
    logger.info(net)

    for param in net.parameters():
        param_count = param.numel()
        num_params += param_count
        if param.requires_grad:
            num_params_train += param_count

    logger.info("Total number of parameters: %d" % num_params)
    logger.info("Total number of trainable parameters: %d" % num_params_train)


def generate_split(
    cls_ids,
    val_num,
    test_num,
    samples,
    n_splits=5,
    seed=7,
    label_frac=1.0,
    custom_test_ids=None,
):
    """Generate train/validation/test splits for cross-validation.
    
    Args:
        cls_ids: List of arrays containing indices for each class.
        val_num: List of validation sample counts per class.
        test_num: List of test sample counts per class.
        samples: Total number of samples.
        n_splits: Number of cross-validation splits (default: 5).
        seed: Random seed for reproducibility (default: 7).
        label_frac: Fraction of training labels to use (default: 1.0).
        custom_test_ids: Optional pre-defined test indices.
        
    Yields:
        Tuple of (train_ids, val_ids, test_ids) for each split.
    """
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(
                cls_ids[c], indices
            )  # all indices of this class
            val_ids = np.random.choice(
                possible_indices, val_num[c], replace=False
            )  # validation ids

            remaining_ids = np.setdiff1d(
                possible_indices, val_ids
            )  # indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None:  # sample test split
                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)

            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    """Return the nth item from an iterator or default if exhausted.
    
    Args:
        iterator: Iterator to consume.
        n: Index of item to return (0-based). If None, exhausts the iterator.
        default: Value to return if iterator is exhausted (default: None).
        
    Returns:
        The nth item from the iterator or default.
    """
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    """Calculate classification error rate.
    
    Args:
        Y_hat: Predicted labels tensor.
        Y: Ground truth labels tensor.
        
    Returns:
        Error rate as a float (1.0 - accuracy).
    """
    error = 1.0 - Y_hat.float().eq(Y.float()).float().mean().item()
    return error


def make_weights_for_balanced_classes_split(dataset):
    """Create sample weights for balanced class sampling.
    
    Args:
        dataset: Dataset with slide_cls_ids attribute and getlabel method.
        
    Returns:
        Tensor of sample weights for balanced sampling.
    """
    num_samples = float(len(dataset))
    num_classes = len(dataset.slide_cls_ids)
    weight_per_class = [
        num_samples / len(dataset.slide_cls_ids[c]) for c in range(num_classes)
    ]
    weight = [0] * int(num_samples)
    for idx in range(len(dataset)):
        class_label = dataset.getlabel(idx)
        weight[idx] = weight_per_class[class_label]

    return torch.DoubleTensor(weight)


def initialize_weights(module):
    """Initialize weights for linear and batch norm layers.
    
    Args:
        module: Neural network module to initialize.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
