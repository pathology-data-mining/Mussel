import argparse
import os
import sys
import time

import h5py
import openslide
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset_h5 import Whole_Slide_Bag_FP
from models.resnet_custom import resnet50_baseline
from utils.file_utils import save_hdf5
from utils.utils import collate_features


def compute_w_loader(
    file_path,
    output_path,
    wsi,
    model,
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

            if args.model == "quilt":
                features = model.encode_image(batch)
            else:
                features = model(batch)
            features = features.cpu().numpy()

            asset_dict = {"features": features, "coords": coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = "a"

    return output_path


parser = argparse.ArgumentParser(description="Feature Extraction")
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--slide_file_path", type=str, default=None)
parser.add_argument("--patch_file_path", type=str, default=None)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--gpus", type=list, default=[0])
args = parser.parse_args()

assert torch.cuda.is_available(), "no cuda available"
device = torch.device(args.gpus[0])

if __name__ == "__main__":
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "pt_files"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "h5_files"), exist_ok=True)

    print("loading model checkpoint")
    if args.model == "resnet50":
        model = resnet50_baseline(pretrained=True)
        preprocessing = None
    elif args.model == "ctranspath":
        sys.path.append("/gpfs/mskmind_ess/boehmk/python_bin/TransPath")
        model = torch.load(
            "/gpfs/mskmind_ess/boehmk/python_bin/TransPath/CTransPath_Model.pt"
        )
        preprocessing = None
    elif args.model == "quilt":
        import open_clip

        model, _, preprocessing = open_clip.create_model_and_transforms(
            "hf-hub:wisdomik/QuiltNet-B-16-PMB"
        )
        tokenizer = open_clip.get_tokenizer("hf-hub:wisdomik/QuiltNet-B-16-PMB")
    else:
        raise ValueError("model not recognized")

    model = model.to(device)
    if len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids=args.gpus)
    model.eval()

    # extract features
    slide_id = os.path.basename(args.slide_file_path).replace(".svs", "")
    output_path = os.path.join(args.save_dir, "h5_files", f"{slide_id}.h5")

    time_start = time.time()
    wsi = openslide.open_slide(args.slide_file_path)
    output_file_path = compute_w_loader(
        args.patch_file_path,
        output_path,
        wsi,
        model=model,
        preprocess=preprocessing,
        batch_size=args.batch_size,
        verbose=1,
        print_every=20,
        use_imagenet_rgb_dist=preprocessing is None,
    )
    time_elapsed = time.time() - time_start
    print(
        "\ncomputing features for {} took {} s".format(output_file_path, time_elapsed)
    )

    file = h5py.File(output_file_path, "r")
    features = file["features"][:]
    print("features size: ", features.shape)
    print("coordinates size: ", file["coords"].shape)
    file.close()

    features = torch.from_numpy(features)
    torch.save(
        features, output_path.replace(".h5", ".pt").replace("h5_files", "pt_files")
    )
