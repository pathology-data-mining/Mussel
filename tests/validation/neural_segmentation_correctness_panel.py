#!/usr/bin/env python
"""Run neural segmentation correctness validation on a fixed multi-slide panel."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tiffslide

from mussel.utils.neural_seg import NeuralTissueSegmenter
from mussel.utils.segment import segment_tissue

DEFAULT_SLIDES = [
    "tests/testdata/948176.svs",
    "docker_work/8471385.svs",
    "/gpfs/mskmind_ess/limr/repos/Mussel-2.old/tests/testdata/1065626.svs",
    "/gpfs/mskmind_ess/limr/repos/workflows2/1005517.svs",
    "/gpfs/mskmind_ess/limr/repos/hf/prov-gigapath/sample_data/PROV-000-000001.ndpi",
    "/gpfs/mskmind_ess/limr/gpu1_cache/mosaic/tcga_slides/d65c5d21-6333-4a9e-9a2a-139a122a3c8a/TCGA-AZ-4313-01Z-00-DX1.5e7ecf69-d1fd-4997-9dcc-ab8e9f10b423.svs",
    "/gpfs/mskmind_ess/limr/gpu1_cache/mosaic/tcga_slides/6a0ea716-a5f2-47f3-880b-537a5cdc2324/TCGA-86-8074-01Z-00-DX1.0c34b434-8701-4060-a4ea-08a72371ee1e.svs",
    "/gpfs/mskmind_ess/limr/repos/Mussel-3/panda_slides/train_images/001d865e65ef5d2579c190a0e0350d8f.tiff",
    "/gpfs/mskmind_ess/limr/repos/Mussel-3/tcga_slides/TCGA-RM-A68W-01Z-00-DX1.4E62E4F4-415C-46EB-A6C8-45BA14E82708.svs",
    "/gpfs/mskmind_ess/limr/repos/mussel-nf/tests/data/1079807.svs",
]


def _slide_metadata(slide_path: str) -> dict[str, Any]:
    with tiffslide.TiffSlide(slide_path) as slide:
        props = slide.properties
        mpp_x = props.get("tiffslide.mpp-x") or props.get("openslide.mpp-x")
        mpp_y = props.get("tiffslide.mpp-y") or props.get("openslide.mpp-y")
        width, height = slide.dimensions
        return {
            "width": int(width),
            "height": int(height),
            "levels": len(slide.level_dimensions),
            "mpp_x": float(mpp_x) if mpp_x is not None else None,
            "mpp_y": float(mpp_y) if mpp_y is not None else None,
        }


def _run_segmentation(
    *,
    slide_path: str,
    slide_id: str,
    seg_model: str,
    output_dir: Path,
    patch_size: int,
    mpp: float,
    tissue_area_threshold: float,
    neural_segmenter: NeuralTissueSegmenter | None,
) -> dict[str, Any]:
    h5_path = output_dir / f"{slide_id}.{seg_model}.h5"
    start = time.monotonic()
    result = segment_tissue(
        slide_path=slide_path,
        slide_id=slide_id,
        patch_size=patch_size,
        mpp=mpp,
        tissue_area_threshold=tissue_area_threshold,
        output_h5_path=str(h5_path),
        seg_model=seg_model,
        neural_segmenter=neural_segmenter,
    )
    elapsed_sec = time.monotonic() - start
    if result is None:
        return {
            "seg_model": seg_model,
            "status": "failed",
            "error": "segment_tissue returned None",
            "elapsed_sec": elapsed_sec,
            "h5_path": str(h5_path),
        }

    _polygon, _grid, coords, attrs = result
    with h5py.File(h5_path, "r") as h5:
        h5_coords = h5["coords"][:]
        h5_attrs = dict(h5["coords"].attrs)

    return {
        "seg_model": seg_model,
        "status": "passed",
        "elapsed_sec": elapsed_sec,
        "h5_path": str(h5_path),
        "patch_count": int(len(coords)),
        "h5_patch_count": int(h5_coords.shape[0]),
        "coords": h5_coords,
        "attrs": attrs,
        "h5_attrs": h5_attrs,
    }


def _validate_result(
    *,
    slide_meta: dict[str, Any],
    seg_model: str,
    result: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if result["status"] != "passed":
        return [result.get("error", f"{seg_model} segmentation failed")]

    if result["patch_count"] <= 0:
        failures.append(f"{seg_model} produced zero patches")
    if result["h5_patch_count"] != result["patch_count"]:
        failures.append(
            f"{seg_model} HDF5 count mismatch: "
            f"{result['h5_patch_count']} != {result['patch_count']}"
        )
    if result["h5_attrs"].get("seg_model") != seg_model:
        failures.append(
            f"{seg_model} HDF5 seg_model attr is "
            f"{result['h5_attrs'].get('seg_model')!r}"
        )

    coords = result["coords"]
    if coords.size:
        if coords.ndim != 2 or coords.shape[1] != 2:
            failures.append(f"{seg_model} coords shape is {coords.shape}")
        elif not (
            np.all(coords[:, 0] >= 0)
            and np.all(coords[:, 1] >= 0)
            and np.all(coords[:, 0] < slide_meta["width"])
            and np.all(coords[:, 1] < slide_meta["height"])
        ):
            failures.append(f"{seg_model} produced out-of-bounds coordinates")

    return failures


def _summarize_for_json(row: dict[str, Any]) -> dict[str, Any]:
    summarized = dict(row)
    for key in ("classic", "neural"):
        if key in summarized:
            summarized[key] = {
                k: v
                for k, v in summarized[key].items()
                if k not in {"coords", "attrs", "h5_attrs"}
            }
    return summarized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slides", nargs="*", default=DEFAULT_SLIDES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--mpp", type=float, default=0.5)
    parser.add_argument("--tissue-area-threshold", type=float, default=1.0)
    parser.add_argument("--ratio-min", type=float, default=0.10)
    parser.add_argument("--ratio-max", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    neural_segmenter = NeuralTissueSegmenter(
        device=args.device,
        batch_size=args.batch_size,
    )
    rows = []
    failures = []

    for index, slide_path in enumerate(args.slides, start=1):
        slide_id = f"{index:02d}_{Path(slide_path).stem}"
        row: dict[str, Any] = {
            "slide_index": index,
            "slide_id": slide_id,
            "slide_path": slide_path,
            "exists": os.path.exists(slide_path),
        }
        print(f"[{index}/{len(args.slides)}] {slide_path}", flush=True)

        try:
            if not row["exists"]:
                raise FileNotFoundError(slide_path)
            row.update(_slide_metadata(slide_path))
            row["classic"] = _run_segmentation(
                slide_path=slide_path,
                slide_id=slide_id,
                seg_model="classic",
                output_dir=output_dir,
                patch_size=args.patch_size,
                mpp=args.mpp,
                tissue_area_threshold=args.tissue_area_threshold,
                neural_segmenter=None,
            )
            row["neural"] = _run_segmentation(
                slide_path=slide_path,
                slide_id=slide_id,
                seg_model="neural",
                output_dir=output_dir,
                patch_size=args.patch_size,
                mpp=args.mpp,
                tissue_area_threshold=args.tissue_area_threshold,
                neural_segmenter=neural_segmenter,
            )

            slide_failures = []
            slide_failures.extend(
                _validate_result(
                    slide_meta=row,
                    seg_model="classic",
                    result=row["classic"],
                )
            )
            slide_failures.extend(
                _validate_result(
                    slide_meta=row,
                    seg_model="neural",
                    result=row["neural"],
                )
            )

            classic_count = row["classic"].get("patch_count", 0)
            neural_count = row["neural"].get("patch_count", 0)
            row["neural_classic_patch_ratio"] = (
                float(neural_count) / float(classic_count) if classic_count else None
            )
            if row["neural_classic_patch_ratio"] is not None and not (
                args.ratio_min <= row["neural_classic_patch_ratio"] <= args.ratio_max
            ):
                slide_failures.append(
                    "neural/classic patch-count ratio "
                    f"{row['neural_classic_patch_ratio']:.3f} outside "
                    f"[{args.ratio_min}, {args.ratio_max}]"
                )

            row["status"] = "passed" if not slide_failures else "failed"
            row["failures"] = slide_failures
        except Exception as exc:
            row["status"] = "failed"
            row["failures"] = [f"{type(exc).__name__}: {exc}"]
            row["traceback"] = traceback.format_exc()

        if row["status"] != "passed":
            failures.append(row["slide_id"])
        rows.append(row)
        summary = _summarize_for_json(row)
        print(json.dumps(summary, indent=2), flush=True)

    json_path = output_dir / "neural_segmentation_correctness_panel.json"
    csv_path = output_dir / "neural_segmentation_correctness_panel.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([_summarize_for_json(row) for row in rows], f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "slide_index",
            "slide_id",
            "status",
            "width",
            "height",
            "levels",
            "mpp_x",
            "mpp_y",
            "classic_patch_count",
            "neural_patch_count",
            "neural_classic_patch_ratio",
            "classic_elapsed_sec",
            "neural_elapsed_sec",
            "failures",
            "slide_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "slide_index": row.get("slide_index"),
                    "slide_id": row.get("slide_id"),
                    "status": row.get("status"),
                    "width": row.get("width"),
                    "height": row.get("height"),
                    "levels": row.get("levels"),
                    "mpp_x": row.get("mpp_x"),
                    "mpp_y": row.get("mpp_y"),
                    "classic_patch_count": row.get("classic", {}).get("patch_count"),
                    "neural_patch_count": row.get("neural", {}).get("patch_count"),
                    "neural_classic_patch_ratio": row.get("neural_classic_patch_ratio"),
                    "classic_elapsed_sec": row.get("classic", {}).get("elapsed_sec"),
                    "neural_elapsed_sec": row.get("neural", {}).get("elapsed_sec"),
                    "failures": "; ".join(row.get("failures", [])),
                    "slide_path": row.get("slide_path"),
                }
            )

    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    if failures:
        print(f"FAILED slides: {', '.join(failures)}", flush=True)
        return 1
    print("All slides passed neural segmentation correctness checks.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
