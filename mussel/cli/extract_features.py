import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import h5py
import hydra
import open_clip
import tiffslide as openslide
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING
from torch.utils.data import DataLoader

from mussel.datasets.h5 import Whole_Slide_Bag_FP
from mussel.models.resnet_custom import resnet50_baseline
from mussel.utils.file import save_hdf5
from mussel.utils.ml import collate_features
from mussel.utils.timer import timed


class ModelType(Enum):
    RESNET50 = 'resnet50'
    CTRANSPATH = 'ctranspath'
    GIGAPATH = 'gigapath'
    CLIP = 'clip'


@dataclass
class ExtractFeaturesConfig:
    patch_h5_path: str = MISSING
    slide_path: str = MISSING
    output_h5_path: str = MISSING
    output_pt_path: str = MISSING
    model_type: ModelType = ModelType.CLIP
    model_path: Optional[str] = "hf-hub:wisdomik/QuiltNet-B-16-PMB"
    batch_size: int = 64
    use_gpu: bool = True
    gpu_device_ids: Optional[List[int]] = field(default_factory=list)
    num_workers: int = 32


@timed
def compute_w_loader(
    file_path,
    output_h5_path,
    wsi_path,
    model_obj,
    model_type: ModelType,
    device,
    batch_size=8,
    verbose=0,
    print_every=20,
    use_imagenet_rgb_dist=True,
    preprocess=None,
    num_workers=32,
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

    dataset = Whole_Slide_Bag_FP(
        file_path=file_path,
        wsi_path=wsi_path,
        use_imagenet_rgb_dist=use_imagenet_rgb_dist,
        preprocess=preprocess,
    )
    kwargs = {"num_workers": num_workers, "pin_memory": True}
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        **kwargs,
        collate_fn=collate_features,
        worker_init_fn=dataset.worker_init,
        shuffle=False,
    )

    if verbose > 0:
        logger.info("processing {}: total of {} batches".format(file_path, len(loader)))

    mode = "w"
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                logger.info(
                    "batch {}/{}, {} files processed".format(
                        count, len(loader), count * batch_size
                    )
                )
            batch = batch.to(device, non_blocking=True)

            if model_type == ModelType.CLIP:
                features = model_obj.encode_image(batch)
            else:
                features = model_obj(batch)
            features = features.cpu().numpy()

            asset_dict = {"features": features, "coords": coords}
            save_hdf5(output_h5_path, asset_dict, attr_dict=None, mode=mode)
            mode = "a"

    return output_h5_path

cs = ConfigStore.instance()
cs.store(name="extract_features_config", node=ExtractFeaturesConfig)

@hydra.main(version_base=None, config_path=".", config_name="extract_features_config")
def main(cfg: ExtractFeaturesConfig):

    device = torch.device("cpu")
    if cfg.use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("cuda not available, using cpu")
    logger.info("loading model checkpoint")
    if cfg.model_type == ModelType.RESNET50:
        model = resnet50_baseline(pretrained=True)
        preprocessing = None
    elif cfg.model_type == ModelType.CTRANSPATH:
        from transpath.ctran import ctranspath
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(cfg.model_path)
        model.load_state_dict(td['model'], strict=True)
        preprocessing = None
    elif cfg.model_type == ModelType.GIGAPATH:
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        preprocessing = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    elif cfg.model_type == ModelType.CLIP:
        model, _, preprocessing = open_clip.create_model_and_transforms(
            cfg.model_path,
        )
    else:
        raise ValueError("model not recognized")

    model = model.to(device)
    if cfg.gpu_device_ids and len(cfg.gpu_device_ids) > 1:
        model = nn.DataParallel(model, device_ids=cfg.gpu_device_ids)
    model.eval()

    # extract features
    output_file_path = compute_w_loader(
        file_path=cfg.patch_h5_path,
        output_h5_path=cfg.output_h5_path,
        wsi_path=cfg.slide_path,
        model_obj=model,
        model_type=cfg.model_type,
        preprocess=preprocessing,
        device=device,
        batch_size=cfg.batch_size,
        verbose=1,
        print_every=20,
        use_imagenet_rgb_dist=preprocessing is None,
        num_workers=cfg.num_workers,
    )

    file = h5py.File(output_file_path, "r")
    features = file["features"][:]
    logger.info(f"features size: {features.shape} ")
    logger.info(f'coordinates size: {file["coords"].shape} ')
    file.close()

    features = torch.from_numpy(features)
    torch.save(
        features, cfg.output_pt_path
    )

if __name__ == "__main__":
    main()
