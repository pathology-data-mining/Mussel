import torch
import torch.nn as nn
import os
import time
from datasets.dataset_h5 import Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import collate_features
from utils.file_utils import save_hdf5
import h5py
import openslide
import sys

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

sys.path.append("/gpfs/mskmind_ess/boehmk/python_bin/TransPath")


def compute_w_loader(
    file_path,
    output_path,
    wsi,
    model,
    batch_size=8,
    verbose=0,
    print_every=20,
    use_imagenet_rgb_dist=True,
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
    )
    x, y = dataset[0]
    kwargs = {"num_workers": 16, "pin_memory": True} if device.type == "cuda" else {}
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features
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
args = parser.parse_args()


if __name__ == "__main__":
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, "pt_files"), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, "h5_files"), exist_ok=True)

    print("loading model checkpoint")
    if args.model == "resnet50":
        model = resnet50_baseline(pretrained=True)
    elif args.model == "ctranspath":
        model = torch.load(
            "/gpfs/mskmind_ess/boehmk/python_bin/TransPath/CTransPath_Model.pt"
        )
    else:
        raise ValueError("model not recognized")

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    # extract features
    slide_id = os.path.basename(args.slide_file_path).replace(".svs", "")
    output_path = os.path.join(args.feat_dir, "h5_files", f"{slide_id}.h5")

    time_start = time.time()
    wsi = openslide.open_slide(args.slide_file_path)
    output_file_path = compute_w_loader(
        args.patch_file_path,
        output_path,
        wsi,
        model=model,
        batch_size=args.batch_size,
        verbose=1,
        print_every=20,
        use_imagenet_rgb_dist=True,
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
