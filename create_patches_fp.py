# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df

# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd


def stitching(file_path, wsi_object, downscale=64, custom_downsample=1):
    start = time.time()
    heatmap = StitchCoords(
        file_path,
        wsi_object,
        downscale=downscale,
        bg_color=(0, 0, 0),
        alpha=-1,
        draw_grid=False,
        custom_downsample=custom_downsample,
    )
    total_time = time.time() - start

    return heatmap, total_time


def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None):
    ### Start Seg Timer
    start_time = time.time()
    # Use segmentation file
    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    # Segment
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(
    source,
    save_dir,
    patch_save_dir,
    mask_save_dir,
    stitch_save_dir,
    patch_size=256,
    step_size=256,
    custom_downsample=1,
    seg_params={
        "seg_level": -1,
        "sthresh": 8,
        "mthresh": 7,
        "close": 4,
        "use_otsu": False,
        "keep_ids": "none",
        "exclude_ids": "none",
    },
    filter_params={"a_t": 100, "a_h": 16, "max_n_holes": 8},
    vis_params={"vis_level": -1, "line_thickness": 100},
    patch_params={"use_padding": True, "contour_fn": "four_pt"},
    patch_level=0,
    use_default_params=True,
    seg=False,
    save_mask=True,
    stitch=False,
    patch=False,
    auto_skip=True,
    process_list=None,
):
    slide_id = os.path.basename(source).replace(".svs", "")
    # Inialize WSI
    WSI_object = WholeSlideImage(source)

    if use_default_params:
        current_vis_params = vis_params.copy()
        current_filter_params = filter_params.copy()
        current_seg_params = seg_params.copy()
        current_patch_params = patch_params.copy()
    else:
        raise NotImplementedError("Custom parameters not yet implemented")

    if current_vis_params["vis_level"] < 0:
        if len(WSI_object.level_dim) == 1:
            current_vis_params["vis_level"] = 0

        else:
            wsi = WSI_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(64)
            current_vis_params["vis_level"] = best_level

    if current_seg_params["seg_level"] < 0:
        if len(WSI_object.level_dim) == 1:
            current_seg_params["seg_level"] = 0

        else:
            wsi = WSI_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(64)
            current_seg_params["seg_level"] = best_level

    keep_ids = str(current_seg_params["keep_ids"])
    if keep_ids != "none" and len(keep_ids) > 0:
        str_ids = current_seg_params["keep_ids"]
        current_seg_params["keep_ids"] = np.array(str_ids.split(",")).astype(int)
    else:
        current_seg_params["keep_ids"] = []

    exclude_ids = str(current_seg_params["exclude_ids"])
    if exclude_ids != "none" and len(exclude_ids) > 0:
        str_ids = current_seg_params["exclude_ids"]
        current_seg_params["exclude_ids"] = np.array(str_ids.split(",")).astype(int)
    else:
        current_seg_params["exclude_ids"] = []

    w, h = WSI_object.level_dim[current_seg_params["seg_level"]]
    if w * h > 1e8:
        print(
            "level_dim {} x {} is likely too large for successful segmentation, aborting".format(
                w, h
            )
        )
        return

    seg_time_elapsed = -1
    if seg:
        WSI_object, seg_time_elapsed = segment(
            WSI_object, current_seg_params, current_filter_params
        )

    if save_mask:
        mask = WSI_object.visWSI(**current_vis_params)
        mask_path = os.path.join(mask_save_dir, slide_id + ".jpg")
        mask.save(mask_path)

    patch_time_elapsed = -1  # Default time
    if patch:
        current_patch_params.update(
            {
                "patch_level": patch_level,
                "patch_size": patch_size,
                "step_size": step_size,
                "save_path": patch_save_dir,
                "custom_downsample": custom_downsample,
            }
        )
        file_path, patch_time_elapsed = patching(
            WSI_object=WSI_object,
            **current_patch_params,
        )

    stitch_time_elapsed = -1
    if stitch:
        file_path = os.path.join(patch_save_dir, slide_id + ".h5")
        if os.path.isfile(file_path):
            heatmap, stitch_time_elapsed = stitching(
                file_path, WSI_object, downscale=64, custom_downsample=custom_downsample
            )
            stitch_path = os.path.join(stitch_save_dir, slide_id + ".jpg")
            heatmap.save(stitch_path)

    print("segmentation took {} seconds".format(seg_time_elapsed))
    print("patching took {} seconds".format(patch_time_elapsed))
    print("stitching took {} seconds".format(stitch_time_elapsed))


parser = argparse.ArgumentParser(description="seg and patch")
parser.add_argument("--source", type=str, help="path to wsi .svs file")
parser.add_argument("--step_size", type=int, default=256, help="step_size")
parser.add_argument("--patch_size", type=int, default=256, help="patch_size")
parser.add_argument("--patch", default=False, action="store_true")
parser.add_argument("--seg", default=False, action="store_true")
parser.add_argument("--stitch", default=False, action="store_true")
parser.add_argument("--save_dir", type=str, help="directory to save processed data")
parser.add_argument(
    "--preset",
    default=None,
    type=str,
    help="predefined profile of default segmentation and filter parameters (.csv)",
)
parser.add_argument(
    "--patch_level", type=int, default=0, help="downsample level at which to patch"
)
parser.add_argument(
    "--custom_downsample",
    type=int,
    choices=[1, 2],
    default=1,
    help="custom downscale when native downsample is not available (only tested w/ 2x downscale)",
)

if __name__ == "__main__":
    args = parser.parse_args()

    patch_save_dir = os.path.join(args.save_dir, "patches")
    mask_save_dir = os.path.join(args.save_dir, "masks")
    stitch_save_dir = os.path.join(args.save_dir, "stitches")

    print("wsi_file: ", args.source)
    print("patch_save_dir: ", patch_save_dir)
    print("mask_save_dir: ", mask_save_dir)
    print("stitch_save_dir: ", stitch_save_dir)

    directories = {
        "source": args.source,
        "save_dir": args.save_dir,
        "patch_save_dir": patch_save_dir,
        "mask_save_dir": mask_save_dir,
        "stitch_save_dir": stitch_save_dir,
    }

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ["source"]:
            os.makedirs(val, exist_ok=True)

    seg_params = {
        "seg_level": -1,
        "sthresh": 8,
        "mthresh": 7,
        "close": 4,
        "use_otsu": False,
        "keep_ids": "none",
        "exclude_ids": "none",
    }
    filter_params = {"a_t": 100, "a_h": 16, "max_n_holes": 8}
    vis_params = {"vis_level": -1, "line_thickness": 250}
    patch_params = {"use_padding": True, "contour_fn": "four_pt"}

    if args.preset:
        preset_df = pd.read_csv(os.path.join("presets", args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {
        "seg_params": seg_params,
        "filter_params": filter_params,
        "patch_params": patch_params,
        "vis_params": vis_params,
    }

    print(parameters)

    seg_and_patch(
        **directories,
        **parameters,
        patch_size=args.patch_size,
        step_size=args.step_size,
        custom_downsample=args.custom_downsample,
        seg=args.seg,
        use_default_params=True,
        save_mask=True,
        stitch=args.stitch,
        patch_level=args.patch_level,
        patch=args.patch,
    )
