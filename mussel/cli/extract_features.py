import os
import pickle
import ssl
from dataclasses import dataclass
from typing import List, Optional

import h5py
import hydra
import tiffslide as openslide
import torch
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from mussel.datasets.h5 import Whole_Slide_Bag_FP
from mussel.models.model_factory import ModelType, get_model_factory
from mussel.utils.file import save_hdf5
from mussel.utils.ml import collate_features
from mussel.utils.timer import timed

ssl._create_default_https_context = ssl._create_unverified_context


@dataclass
class ExtractFeaturesConfig:
    patch_h5_path: str = MISSING
    slide_path: str = MISSING
    output_h5_path: str = MISSING
    output_pt_path: str = MISSING
    model_type: ModelType = ModelType.CLIP
    model_path: Optional[str] = None
    patch_path: Optional[str] = None
    batch_size: int = 64
    use_gpu: bool = True
    gpu_device_id: Optional[int] = None
    gpu_device_ids: Optional[List[int]] = None
    num_workers: int = 32


@timed
def compute_w_loader(
    file_path,
    output_h5_path,
    wsi_path,
    model_fun,
    patch_path=None,
    batch_size=64,
    verbose=0,
    print_every=20,
    use_imagenet_rgb_dist=True,
    preprocess=None,
    num_workers=32,
    pin_memory=True,
):
    """
    args:
            file_path: directory of bag (.h5 file)
            output_h5_path: file path to save computed features (.h5 file)
            model_type: model type
            batch_size: batch_size for computing features in batches
            verbose: level of feedback
            pretrained: use weights pretrained on imagenet
    """

    # if patch_path is a directory, assume it is a directory of pre-tiled images
    # that can be processed independently and collated as-needed.
    if patch_path is not None and os.path.isdir(patch_path):

        dataset = ImageFolder(
            root=patch_path,
            transform=preprocess,
        )

        # override batch_size for filepath based feature extraction
        batch_size = 1

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=None,
            worker_init_fn=None,
            shuffle=False,
        )
    else:

        dataset = Whole_Slide_Bag_FP(
            file_path=file_path,
            wsi_path=wsi_path,
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
        logger.info("processing {}: total of {} batches".format(file_path, len(loader)))

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
        if patch_path is not None and os.path.isdir(patch_path):
            asset_dict = {"features": features}
            fname = os.path.splitext(os.path.basename(dataset.imgs[count][0]))[0]
            save_hdf5(
                os.path.join(output_h5_path, f"{fname}.h5"),
                asset_dict,
                attr_h5_path=None,
                mode=mode,
            )
        else:
            asset_dict = {"features": features, "coords": coords}
            save_hdf5(output_h5_path, asset_dict, attr_h5_path=file_path, mode=mode)
        mode = "a"

    return output_h5_path


cs = ConfigStore.instance()
cs.store(name="extract_features_config", node=ExtractFeaturesConfig)


@hydra.main(version_base=None, config_path=".", config_name="extract_features_config")
def main(cfg: ExtractFeaturesConfig):

    gpu_device_id = cfg.gpu_device_id
    if cfg.gpu_device_ids:
        gpu_device_id = cfg.gpu_device_ids

    logger.info("loading model checkpoint")

    if cfg.model_path is None:
        cfg.model_path = cfg.model_type.hf_path

    model_obj = None
    if cfg.model_path.endswith(".pkl"):
        with open(cfg.model_path, "rb") as f:
            model_obj = pickle.load(f)

    model_factory = get_model_factory(cfg.model_type)
    if model_factory is None:
        raise ValueError("model not recognized")
    model = model_factory.get_model(
        cfg.model_path, model_obj, cfg.use_gpu, gpu_device_id
    )
    preprocessing = model.get_preprocessing_fun()

    # extract features
    output_file_path = compute_w_loader(
        file_path=cfg.patch_h5_path,
        output_h5_path=cfg.output_h5_path,
        wsi_path=cfg.slide_path,
        model_fun=model.get_model_fun(),
        preprocess=preprocessing,
        pin_memory=cfg.use_gpu,
        patch_path=cfg.patch_path,
        batch_size=cfg.batch_size,
        verbose=1,
        print_every=20,
        use_imagenet_rgb_dist=preprocessing is None,
        num_workers=cfg.num_workers,
    )

    if output_file_path is None:
        logger.info("No features found")
        return

    # TODO: potentially output .pt files for folder based feature extraction if needed
    if cfg.patch_path is not None and os.path.isdir(cfg.patch_path):
        return
    else:
        file = h5py.File(output_file_path, "r")
        features = file["features"][:]
        logger.info(f"features size: {features.shape} ")
        # logger.info(f'coordinates size: {file["coords"].shape} ')
        file.close()

        features = torch.from_numpy(features)
        torch.save(features, cfg.output_pt_path)


if __name__ == "__main__":
    main()
