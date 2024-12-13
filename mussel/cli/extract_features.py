import argparse
import os
import pickle
import ssl
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
import timm
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import MISSING
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from mussel.datasets.h5 import Whole_Slide_Bag_FP
from mussel.models.resnet_custom import resnet50_baseline
from mussel.utils.file import save_hdf5
from mussel.utils.ml import collate_features
from mussel.utils.timer import timed

ssl._create_default_https_context = ssl._create_unverified_context


class ModelType(Enum):
    def __init__(self, id, code, hf_path):
        self.id = id
        self.code = code
        self.hf_path = hf_path

    RESNET50 = 1, "resnet50", ""
    CTRANSPATH = 2, "ctranspath", ""
    GIGAPATH = 3, "gigapath", "hf-hub:prov-gigapath/prov-gigapath"
    VIRCHOW = 4, "virchow", "hf-hub:paige-ai/Virchow"
    OPTIMUS = 5, "optimus", "hf-hub:bioptimus/H-optimus-0"
    CLIP = 6, "clip", "hf-hub:wisdomik/QuiltNet-B-16-PMB"


@dataclass
class ExtractFeaturesConfig:
    patch_h5_path: str = MISSING
    slide_path: str = MISSING
    output_h5_path: str = MISSING
    output_pt_path: str = MISSING
    model_type: ModelType = ModelType.CLIP
    model_path: Optional[str] = None
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
    device_type,
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

    # if file_path is a directory, assume it is a directory of pre-tiled images
    # that can be processed independently and collated as-needed.
    if os.path.isdir(wsi_path):
        logger.info(wsi_path)

        dataset = ImageFolder(
            root=wsi_path,
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
        with torch.no_grad(), torch.inference_mode(), torch.autocast(
            device_type=device_type, dtype=torch.float16
        ):
            if count % print_every == 0:
                logger.info(
                    "batch {}/{}, {} tiles processed".format(
                        count, len(loader), count * batch_size
                    )
                )
            batch = batch.to(device, non_blocking=True)

            if model_type == ModelType.CLIP:
                features = model_obj.encode_image(batch)
            else:
                features = model_obj(batch)
            features = features.cpu().numpy()

            if os.path.isdir(wsi_path):
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

    device_type = "cpu"
    pin_memory = False
    if cfg.use_gpu:
        if torch.cuda.is_available():
            pin_memory = True
            device_type = "cuda"
        else:
            logger.warning("cuda not available, using cpu")
    logger.info("loading model checkpoint")
    model = None
    if cfg.model_path is None:
        cfg.model_path = cfg.model_type.hf_path
    if cfg.model_path.endswith(".pkl"):
        with open(cfg.model_path, "rb") as f:
            model = pickle.load(f)
    if cfg.model_type == ModelType.RESNET50:
        model = resnet50_baseline(pretrained=True)
        preprocessing = None
    elif cfg.model_type == ModelType.CTRANSPATH:
        from transpath.ctran import ctranspath

        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(cfg.model_path)
        model.load_state_dict(td["model"], strict=True)
        # ctranspath() module has required torch transforms built in so
        # preprocessing should be None here
        preprocessing = None
    elif cfg.model_type == ModelType.GIGAPATH:
        if model is None:
            model = timm.create_model(cfg.model_path, pretrained=True)
        preprocessing = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    elif cfg.model_type == ModelType.VIRCHOW:
        # need to specify MLP layer and activation function for proper init
        if model is None:
            model = timm.create_model(
                cfg.model_path,
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        preprocessing = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
    elif cfg.model_type == ModelType.CLIP:
        model, _, preprocessing = open_clip.create_model_and_transforms(
            cfg.model_path,
        )
    elif cfg.model_type == ModelType.OPTIMUS:
        if model is None:
            model = timm.create_model(
                cfg.model_path,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False,
            )

        preprocessing = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
    else:
        raise ValueError("model not recognized")

    device = torch.device(device_type)
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
        device_type=device_type,
        pin_memory=pin_memory,
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
    if os.path.isdir(cfg.slide_path):
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
