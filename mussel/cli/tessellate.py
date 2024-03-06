# internal imports
import argparse

# other imports
import os
import time

import numpy as np
import pandas as pd

from mussel.WholeSlideImage import WholeSlideImage
from mussel.utils.wsi import StitchCoords


def stitching(file_path, wsi_object, downscale=64, mpp=0.5):
    start = time.time()
    heatmap = StitchCoords(
        file_path,
        wsi_object,
        downscale=downscale,
        bg_color=(0, 0, 0),
        alpha=-1,
        draw_grid=False,
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
    _ = WSI_object.process_contours(**kwargs)

    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return patch_time_elapsed


def seg_and_patch(
    slide_file_path,
    patch_save_path,
    mask_save_path,
    stitch_save_path,
    patch_size=256,
    step_size=256,
    mpp=0.5,
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
    patch_params={"use_padding": True},
    use_default_params=True,
    seg=False,
    save_mask=True,
    stitch=False,
    patch=False,
):
    # Inialize WSI
    WSI_object = WholeSlideImage(slide_file_path)

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
        mask.save(mask_save_path)

    patch_time_elapsed = -1  # Default time
    if patch:
        current_patch_params.update(
            {
                "mpp": mpp,
                "patch_size": patch_size,
                "step_size": step_size,
                "save_path": patch_save_path,
            }
        )
        patch_time_elapsed = patching(
            WSI_object=WSI_object,
            **current_patch_params,
        )

    stitch_time_elapsed = -1
    if stitch:
        heatmap, stitch_time_elapsed = stitching(
            patch_save_path, WSI_object, downscale=64, mpp=mpp
        )
        heatmap.save(stitch_save_path)


    print("segmentation took {} seconds".format(seg_time_elapsed))
    print("patching took {} seconds".format(patch_time_elapsed))
    print("stitching took {} seconds".format(stitch_time_elapsed))


def main(in_path_wsi,
         out_path_patch,
         out_path_mask,
         out_path_stitch,
         patch_size,
         step_size,
         mpp,

         preset=None):
    # check that slide file exists
    assert os.path.exists(in_path_wsi), f"file {in_path_wsi} does not exist"

    # check file extensions
    assert in_path_wsi.endswith('.svs'), f"file {in_path_wsi} is not a .svs file"
    assert out_path_patch.endswith('.h5'), f"file {out_path_patch} is not a .h5 file"
    assert out_path_mask.endswith('.jpg'), f"file {out_path_mask} is not a .jpg file"
    assert out_path_stitch.endswith('.jpg'), f"file {out_path_stitch} is not a .jpg file"

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
    vis_params = {"vis_level": -1, "line_thickness": 100}
    patch_params = {"use_padding": True}

    if preset:
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

    paths = {
        "slide_file_path": in_path_wsi,
        "patch_save_path": out_path_patch,
        "mask_save_path": out_path_mask,
        "stitch_save_path": out_path_stitch,
    }

    seg_and_patch(
        **paths,
        **parameters,
        patch_size=patch_size,
        step_size=step_size,
        mpp=mpp,
        seg=True,
        use_default_params=True,
        save_mask=True,
        stitch=True,
        patch=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "slide_file_path",
        type=str,
        help="Path to the slide file",
    )
    parser.add_argument(
        "out_path_patch",
        type=str,
        help="Path to save the patching",
    )
    parser.add_argument(
        "out_path_mask",
        type=str,
        help="Path to save the mask",
    )
    parser.add_argument(
        "out_path_stitch",
        type=str,
        help="Path to save the stitched image",
    )
    parser.add_argument( "--patch_size", type=int, default=224, help="Size of each patch")
    parser.add_argument( "--step_size", type=int, default=224, help="Step size between patches")
    parser.add_argument( "--mpp", type=float, default=0.5, help="Microns per pixel")

    args = parser.parse_args()
    args = vars(args)
    main(**args)
