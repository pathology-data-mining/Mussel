import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from mussel.datasets import WholeSlideImageTileDataset
from mussel.models import ModelType, get_model_factory

from .file import save_hdf5
from .ml import collate_features
from .timer import timed

logger = logging.getLogger(__name__)


def extract_features(
    slide_path,
    gpu_device_id,
    model_type,
    model_path,
    use_gpu,
    output_h5_path,
    output_pt_path,
    patch_h5_path=None,
    patch_path=None,
    model_save_path=None,
    batch_size=64,
    num_workers=16,
    gpu_device_ids=None,
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
        extract_features_from_patch_dir(
            patch_path=patch_path,
            output_h5_path=output_h5_path,
            model_fun=model.get_model_fun(),
            preprocess=preprocessing,
            pin_memory=use_gpu,
            batch_size=batch_size,
            verbose=1,
            print_every=20,
            num_workers=num_workers,
        )
    elif patch_h5_path:
        extract_features_from_patch_h5(
            patch_h5_path=patch_h5_path,
            output_h5_path=output_h5_path,
            slide_path=slide_path,
            model_fun=model.get_model_fun(),
            preprocess=preprocessing,
            pin_memory=use_gpu,
            batch_size=batch_size,
            verbose=1,
            print_every=20,
            use_imagenet_rgb_dist=preprocessing is None,
            num_workers=num_workers,
        )
    else:
        raise ValueError("Either patch_path or patch_h5_path must be provided")

    with h5py.File(output_h5_path, "r") as file:
        features = file["features"][:]
        logger.info(f"features size: {features.shape} ")
        # logger.info(f'coordinates size: {file["coords"].shape} ')

        features = torch.from_numpy(features)
        torch.save(features, output_pt_path)


@timed
def extract_features_from_patch_dir(
    patch_path,
    output_h5_path,
    model_fun,
    batch_size=64,
    verbose=0,
    print_every=20,
    preprocess=None,
    num_workers=16,
    pin_memory=True,
):

    dataset = ImageFolder(
        root=patch_path,
        transform=preprocess,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=None,
        shuffle=False,
    )

    if verbose > 0:
        logger.info(
            "processing {}: total of {} batches".format(patch_path, len(loader))
        )

    if len(loader) == 0:
        return None

    mode = "w"

    for count, (batch, labels) in enumerate(loader):
        labels = labels.numpy()
        if count % print_every == 0:
            logger.info(
                "batch {}/{}, {} tiles processed".format(
                    count, len(loader), count * batch_size
                )
            )

        features = model_fun(batch)

        features = features.numpy()
        asset_dict = {
            "features": features,
            "class": labels,
            "image_paths": np.array([x[0] for x in dataset.imgs]).astype("T"),
            "class_to_idx": np.array(
                [np.asarray([k, v], dtype="T") for k, v in dataset.class_to_idx.items()]
            ),
        }
        save_hdf5(output_h5_path, asset_dict, attr_h5_path=None, mode=mode)
        mode = "a"


@timed
def extract_features_from_patch_h5(
    patch_h5_path,
    output_h5_path,
    slide_path,
    model_fun,
    patch_path=None,
    batch_size=64,
    verbose=0,
    print_every=20,
    use_imagenet_rgb_dist=True,
    preprocess=None,
    num_workers=16,
    pin_memory=True,
):
    """
    args:
            patch_h5_path: directory of bag (.h5 file)
            output_h5_path: file path to save computed features (.h5 file)
            model_type: model type
            batch_size: batch_size for computing features in batches
            verbose: level of feedback
            pretrained: use weights pretrained on imagenet
    """

    dataset = WholeSlideImageTileDataset(
        h5_path=patch_h5_path,
        slide_path=slide_path,
        use_imagenet_rgb_dist=use_imagenet_rgb_dist,
        preprocess=preprocess,
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

    if verbose > 0:
        logger.info(
            "processing {}: total of {} batches".format(patch_h5_path, len(loader))
        )

    if len(loader) == 0:
        return None

    mode = "w"

    for count, (batch, coords) in enumerate(loader):
        if count % print_every == 0:
            logger.info(
                "batch {}/{}, {} tiles processed".format(
                    count, len(loader), count * batch_size
                )
            )

        features = model_fun(batch)

        features = features.numpy()
        asset_dict = {"features": features, "coords": coords}
        save_hdf5(output_h5_path, asset_dict, attr_h5_path=patch_h5_path, mode=mode)
        mode = "a"

    return output_h5_path
