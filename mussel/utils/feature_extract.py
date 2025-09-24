import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from functools import singledispatch

from mussel.datasets import WholeSlideImageH5Dataset, WholeSlideImageTileCoordDataset
from mussel.models import ModelType, get_model_factory

from .file import save_hdf5
from .ml import collate_features
from .timer import timed

logger = logging.getLogger(__name__)

@singledispatch
def process_dataset(
    dataset,
    loader,
    model_fun,
    patch_h5_path=None,
    output_h5_path=None,
    print_every=20,
):
    """
    Args:
            dataset: dataset object
            loader: dataloader object
            model_fun: function to extract features from a batch of images
            patch_h5_path: path to the h5 file containing patch coordinates (if any)
            output_h5_path: path to save the extracted features (if any)
            print_every: print progress every n batches
    """
    pass

@process_dataset.register(WholeSlideImageTileCoordDataset)
def _(dataset: WholeSlideImageTileCoordDataset, loader, model_fun, print_every=20):
    all_features = []
    all_labels = []
    for count, (batch, labels) in enumerate(loader):
        if count % print_every == 0:
            logger.info(
                "batch {}/{} tiles processed".format(
                    count, len(loader)
                )
            )

        features = model_fun(batch)
        all_features.append(features.numpy())
        all_labels.append(labels)
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels

@process_dataset.register(ImageFolder)
def _(dataset: ImageFolder, loader, model_fun, patch_h5_path=None, output_h5_path=None, print_every=20):
    asset_dict = {
        "image_paths": np.array([x[0] for x in dataset.imgs]).astype("T"),
        "class_to_idx": np.array(
            [np.asarray([k, v], dtype="T") for k, v in dataset.class_to_idx.items()]
        ),
    }
    save_hdf5(output_h5_path, asset_dict, attr_h5_path=None, mode="w")
    for count, (batch, labels) in enumerate(loader):
        labels = labels.numpy()
        if count % print_every == 0:
            logger.info(
                "batch {}/{} processed".format(
                    count, len(loader)
                )
            )

        features = model_fun(batch)
        features = features.numpy()
        asset_dict = {
            "features": features,
            "class": labels,
        }
        save_hdf5(output_h5_path, asset_dict, attr_h5_path=None, mode="a")
    return output_h5_path

@process_dataset.register(WholeSlideImageH5Dataset)
def _(dataset: WholeSlideImageH5Dataset, loader, model_fun, patch_h5_path=None, output_h5_path=None, print_every=20):
    mode = "w"
    for count, (batch, coords) in enumerate(loader):
        if count % print_every == 0:
            logger.info(
                "batch {}/{} tiles processed".format(
                    count, len(loader)
                )
            )

        features = model_fun(batch)
        features = features.numpy()
        asset_dict = {"features": features, "coords": coords}
        save_hdf5(output_h5_path, asset_dict, attr_h5_path=patch_h5_path, mode=mode)
        mode = "a"
    return output_h5_path



def get_features(
    coords,
    slide_path,
    attrs,
    model_type=ModelType.CLIP,
    model_path=None,
    batch_size=64,
    use_gpu=True,
    gpu_device_id=None,
    gpu_device_ids=None,
    pin_memory=True,
    num_workers=16,
    print_every=20,
):
    logger.info("loading model checkpoint")

    if gpu_device_ids:
        gpu_device_id = gpu_device_ids

    model_factory = get_model_factory(model_type)
    if model_factory is None:
        raise ValueError("model not recognized")
    model = model_factory.get_model(model_path, use_gpu, gpu_device_id)
    preprocessing = model.get_preprocessing_fun()

    dataset = WholeSlideImageTileCoordDataset(
        coords=coords,
        attrs=attrs,
        slide_path=slide_path,
        use_imagenet_rgb_dist=preprocessing is None,
        preprocess=preprocessing
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=dataset.worker_init,
        collate_fn=collate_features,
        shuffle=False,
    )

    features, labels = process_dataset(dataset, loader, model_fun=model.get_model_fun(), print_every=print_every)

    return features, labels


@timed
def save_features(
    patch_h5_path,
    slide_path,
    output_h5_path,
    output_pt_path=None,
    model_type=ModelType.CLIP,
    model_path=None,
    model_save_path=None,
    patch_path=None,
    batch_size=64,
    use_gpu=True,
    gpu_device_id=None,
    gpu_device_ids=None,
    num_workers=16,
    pin_memory=True,
    print_every=20,
):

    if gpu_device_ids:
        gpu_device_id = gpu_device_ids

    logger.info("loading model checkpoint")

    model_factory = get_model_factory(model_type)
    if model_factory is None:
        raise ValueError("model not recognized")
    model = model_factory.get_model(model_path, use_gpu, gpu_device_id)
    if model_save_path is not None:
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"saving model to {model_save_path}")
        model.save(model_save_path)
    preprocessing = model.get_preprocessing_fun()

    if patch_path:
        dataset = ImageFolder(
            root=patch_path,
            transform=preprocessing,
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=None,
            shuffle=False,
        )
    elif patch_h5_path:
        dataset = WholeSlideImageH5Dataset(
            h5_path=patch_h5_path,
            slide_path=slide_path,
            preprocess=preprocessing,
            use_imagenet_rgb_dist=preprocessing is None,
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_features,
            worker_init_fn=dataset.worker_init,
            shuffle=False,
        )
    else:
        raise ValueError("Either patch_path or patch_h5_path must be provided")

    process_dataset(dataset, loader, model_fun=model.get_model_fun(), patch_h5_path=patch_h5_path, output_h5_path=output_h5_path, print_every=print_every)

    if output_pt_path is not None:
        with h5py.File(output_h5_path, "r") as file:
            features = file["features"][:]
            logger.info(f"features size: {features.shape} ")
            # logger.info(f'coordinates size: {file["coords"].shape} ')

            features = torch.from_numpy(features)
            torch.save(features, output_pt_path)


@timed
def filter_features(
    features: torch.Tensor,
    coords,
    classifier,
    threshold: float,
):
    logger.info("Predicting probabilities...")
    inclusion_mask = classifier.predict_proba(features)[:, 1] > threshold
    logger.info(f"{sum(inclusion_mask)} tiles above {threshold} threshold")
    return features[inclusion_mask], np.array(coords)[inclusion_mask]

