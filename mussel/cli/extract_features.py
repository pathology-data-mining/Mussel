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

from mussel.datasets.h5 import Whole_Slide_Bag_FP
from mussel.models.resnet_custom import resnet50_baseline
from mussel.utils.file import save_hdf5
from mussel.utils.ml import collate_features
from mussel.utils.timer import timed


@timed
def compute_w_loader(
    file_path,
    output_path,
    wsi,
    model,
    model_name,
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
            output_path: directory to save computed features (.h5 file)
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

            if model_name == "quilt":
                features = model.encode_image(batch)
            else:
                features = model(batch)
            features = features.cpu().numpy()

            asset_dict = {"features": features, "coords": coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = "a"

    return output_path


def main(h5_feats_path: str,
         pt_feats_path: str,
         patch_path: str,
         slide_path: str,
         transpath_path: Optional[str] = None,
         model_name: str = "resnet50",
         model_path: Optional[str] = None,
         batch_size: int = 64,
         use_gpu: bool = True,
         gpu_device_ids: List[int] = [0]):

    device = torch.device("cpu")
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warn("cuda not available, using cpu")
    logger.info("loading model checkpoint")
    if model_name == "resnet50":
        model = resnet50_baseline(pretrained=True)
        preprocessing = None
    elif model_name == "ctranspath":
        sys.path.append(transpath_path)
        model = torch.load(model_path)
        preprocessing = None
    elif model_name == "quilt":
        model, _, preprocessing = open_clip.create_model_and_transforms(
            "hf-hub:wisdomik/QuiltNet-B-16-PMB"
        )
    elif model_name == "open_clip":
        model, _, preprocessing = open_clip.create_model_and_transforms(
            model_path
        )
    else:
        raise ValueError("model not recognized")

    model = model.to(device)
    if len(gpu_device_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_device_ids)
    model.eval()

    # extract features
    wsi = openslide.open_slide(slide_path)
    output_file_path = compute_w_loader(
        patch_path,
        h5_feats_path,
        wsi,
        model=model,
        model_name=model_name,
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
    fire.Fire(main)
