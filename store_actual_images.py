"""
Inputs: slide_file_path, patch_file_path
Results in .pt file with N_tiles x 3 x img_size x img_size tensor
"""

import argparse
from torch.utils.data import DataLoader
import torch
from datasets.dataset_h5 import Whole_Slide_Bag_FP
from utils.utils import collate_features
import openslide
import time

parser = argparse.ArgumentParser(description="store actual images")
parser.add_argument("--slide_file_path", type=str, default=None)
parser.add_argument("--patch_file_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    time_start = time.time()
    wsi = openslide.open_slide(args.slide_file_path)
    dataset = Whole_Slide_Bag_FP(
        file_path=args.patch_file_path,
        wsi=wsi,
        use_imagenet_rgb_dist=True,
    )
    kwargs = {"num_workers": 32, "pin_memory": True}
    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        **kwargs,
        collate_fn=collate_features,
        shuffle=False,
    )
    with torch.no_grad():
        batch_list = []
        for count, (batch, coords) in enumerate(loader):
            if count % 100 == 0:
                print(
                    "batch {}/{}, {} files processed".format(
                        count, len(loader), count * 32
                    )
                )
            batch_list.append(batch)
        all_tiles = torch.cat(batch_list, dim=0)
    time_elapsed = time.time() - time_start
    print("\ncaching tiles for {} took {} s".format(args.output_path, time_elapsed))
    print(f"all_tiles shape: {all_tiles.shape}")
    torch.save(all_tiles, args.output_path)
    print(f"saved to {args.output_path}")
