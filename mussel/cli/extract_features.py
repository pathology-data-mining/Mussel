import argparse
import os
import sys
import time

import h5py
import openslide
import open_clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import fire
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path

from mussel.datasets.h5 import Whole_Slide_Bag_FP
from mussel.models.resnet_custom import resnet50_baseline
from mussel.utils.file import save_hdf5
from mussel.utils.ml import collate_features
from mussel.utils.timer import timed
from mussel.utils.config import ExtractFeaturesConfig, Model

class Model(Enum):
    RESNET50 = 'resnet50'
    CTRANSPATH = 'ctranspath'
    QUILTNET = 'quiltnet'
    OPENCLIP = 'openclip'

@dataclass
class ExtractFeaturesConfig:
    patch_path: str
    output_path: str
    slide_path: str
    transpath_dir: Optional[str] = None
    model: Model = Model.QUILTNET #y
    model_path: str = "hf-hub:wisdomik/QuiltNet-B-16-PMB"
    batch_size: int = 64
    use_gpu: bool = True
    gpu_device_ids: List[int] = field(default_factory=list)


@timed
def compute_w_loader(
    file_path,
    output_path,
    wsi,
    model_obj,
    model: Model,
    device,
    batch_size=8,
    verbose=0,
    print_every=20,
    use_imagenet_rgb_dist=True,
    preprocess=None,
):
    """
    args:
            file_path: directory of bag (.h5 file)
            output_path: file path to save computed features (.h5 file)
            model: pytorch model
            batch_size: batch_size for computing features in batches
            verbose: level of feedback
            pretrained: use weights pretrained on imagenet
    """

    dataset = Whole_Slide_Bag_FP(
        file_path=file_path,
        wsi=wsi,
        use_imagenet_rgb_dist=use_imagenet_rgb_dist,
        preprocess=preprocess,
    )
    x, y = dataset[0]
    kwargs = {"num_workers": 32, "pin_memory": True}
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        **kwargs,
        collate_fn=collate_features,
        shuffle=False,
    )

    if verbose > 0:
        print("processing {}: total of {} batches".format(file_path, len(loader)))

    mode = "w"
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print(
                    "batch {}/{}, {} files processed".format(
                        count, len(loader), count * batch_size
                    )
                )
            batch = batch.to(device, non_blocking=True)

            if model == Model.QUILTNET:
                features = model.encode_image(batch)
            else:
                features = model(batch)
            features = features.cpu().numpy()

            asset_dict = {"features": features, "coords": coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = "a"

    return output_path


@hydra.main(config_path=".", config_name="extract_features_config")
def extract_features(cfg: ExtractFeaturesConfig):

    device = torch.device("cpu")
    if cfg.use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warn("cuda not available, using cpu")
    logger.info("loading model checkpoint")
    if cfg.model == Model.RESNET50:
        model = resnet50_baseline(pretrained=True)
        preprocessing = None
    elif cfg.model == Model.CTRANSPATH:
        sys.path.append(cfg.transpath_path)
        model = torch.load(cfg.model_path)
        preprocessing = None
    elif cfg.model == Model.QUILTNET:
        model, _, preprocessing = open_clip.create_model_and_transforms(
            cfg.model_path,
        )
    elif cfg.model == Model.OPENCLIP:
        model, _, preprocessing = open_clip.create_model_and_transforms(
            cfg.model_path,
        )
    else:
        raise ValueError("model not recognized")

    model = model.to(device)
    if len(gpu_device_ids) > 1:
        model = nn.DataParallel(model, device_ids=cfg.gpu_device_ids)
    model.eval()

    # extract features
    wsi = openslide.open_slide(cfg.slide_path)
    output_file_path = compute_w_loader(
        cfg.patch_path,
        cfg.output_path,
        wsi,
        model_obj=model,
        model=cfg.model,
        preprocess=preprocessing,
        device=device,
        batch_size=batch_size,
        verbose=1,
        print_every=20,
        use_imagenet_rgb_dist=preprocessing is None,
    )

    file = h5py.File(output_file_path, "r")
    features = file["features"][:]
    print("features size: ", features.shape)
    print("coordinates size: ", file["coords"].shape)
    file.close()

    features = torch.from_numpy(features)
    torch.save(
        features, pt_feats_path
    )

if __name__ == "__main__":
    extract_features()
