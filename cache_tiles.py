"""
Inputs: slide_file_path, patch_file_path, annot_path
Results in .pt file with N_tiles x 3 x img_size x img_size tensor
"""

import argparse
from torch.utils.data import DataLoader
import torch
from datasets.dataset_h5 import Whole_Slide_Bag_FP
from utils.utils import collate_features
import openslide
import time
import pandas as pd
import json


def main(slide_file_path, patch_file_path, output_path, limit_to_class=None, annot_path=None, cache_tile_indices_path=None):
    time_start = time.time()
    if limit_to_class is not None:
        annot = pd.read_csv(annot_path)
        annot['class'] = annot.idxmax(axis=1)
        indices = annot[annot['class'] == limit_to_class].index.tolist()
        print(f"limiting to class {limit_to_class} with {len(indices)} tiles")
    
    wsi = openslide.open_slide(slide_file_path)
    dataset = Whole_Slide_Bag_FP(
        file_path=patch_file_path,
        wsi=wsi,
        use_imagenet_rgb_dist=True,
        limit_to_indices=indices if limit_to_class else None,
    )
    kwargs = {"num_workers": 8, "pin_memory": True}
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
    print("\ncaching tiles for {} took {} s".format(output_path, time_elapsed))
    print(f"all_tiles shape: {all_tiles.shape}")
    torch.save(all_tiles, output_path)
    print(f"saved to {output_path}")
    # save indices as json
    with open(cache_tile_indices_path, "w") as f:
        json.dump(indices, f)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="store actual images")
    PARSER.add_argument("--slide_file_path", type=str, default=None)
    PARSER.add_argument("--patch_file_path", type=str, default=None)
    PARSER.add_argument("--output_path", type=str, default=None)
    PARSER.add_argument("--limit_to_class", type=str, default=None)
    PARSER.add_argument("--annot_path", type=str, default=None)
    PARSER.add_argument("--cache_tile_indices_path", type=str, default=None)
    ARGS = PARSER.parse_args()
    main(ARGS.slide_file_path, ARGS.patch_file_path, ARGS.output_path, ARGS.limit_to_class, ARGS.annot_path, ARGS.cache_tile_indices_path)
