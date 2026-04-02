import base64
import json
import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

import h5py
import hydra
import pandas as pd
import tiffslide as openslide
import torch
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

logger = logging.getLogger(__name__)


@dataclass
class AnnotateConfig:
    """
    features_pt_path (str): Path to the precomputed features in PyTorch format.
    output_csv_path (str): Path to save the classification results in CSV format.
    class_embedding_pt_path (str): Path to the precomputed class embeddings in PyTorch format.
    classes (List[str]): List of class names for zero-shot classification.
    interrogate (bool): Whether to generate an interrogation report.
    slide_path (Optional[str]): Path to the whole slide image for interrogation.
    patch_path (Optional[str]): Path to the HDF5 file containing patch coordinates for interrogation.
    interrogation_report_path (Optional[str]): Path to save the interrogation report HTML file.
    """

    features_pt_path: str = MISSING
    output_csv_path: str = MISSING
    class_embedding_pt_path: str = MISSING
    classes: List[str] = MISSING
    interrogate: bool = False
    slide_path: Optional[str] = None
    patch_path: Optional[str] = None
    interrogation_report_path: Optional[str] = None


def interrogate_function(slide_path, patch_path, interrogation_report_path, df):
    """Generate an HTML interrogation report showing tiles grouped by class.

    Args:
        slide_path: Path to the whole slide image.
        patch_path: Path to the HDF5 file containing patch coordinates.
        interrogation_report_path: Path to save the HTML report.
        df: DataFrame containing classification results with 'class' column.
    """
    slide = openslide.OpenSlide(slide_path)

    with h5py.File(patch_path, "r") as f:
        patch_size = f["coords"].attrs["patch_size"]
        patch_level = f["coords"].attrs["patch_level"]
        logger.info(len(f["coords"]))
        assert len(f["coords"]) == len(
            df
        ), f"{len(f['coords'])} vs {len(df)} tiles, aborting"
        coords = f["coords"][:]

    df["tile_index"] = df.index

    html_content = ""
    for class_name, sub_df in df.groupby("class"):
        html_content += f"<h2>{class_name}</h2>"
        for _, row in sub_df.iterrows():
            tile_index = row["tile_index"]
            tile_coords = coords[tile_index]
            tile = slide.read_region(tile_coords, patch_level, (patch_size, patch_size))
            tile = tile.resize((200, 200))
            tile_bytes = BytesIO()
            tile.save(tile_bytes, format="PNG")
            tile_base64 = base64.b64encode(tile_bytes.getvalue()).decode("utf-8")
            html_content += f"<img src='data:image/png;base64,{tile_base64}' width='200' height='200'>"

    html_document = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interrogation Report</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    with open(interrogation_report_path, "w") as f:
        f.write(html_document)


desc_doc = """== ${hydra.help.app_name} ==
This script performs zero-shot classification on specified classes.
"""

parameter_doc = f"""
== Available Parameters ==
{AnnotateConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="annotate_config", node=AnnotateConfig)


@hydra.main(config_path=".", config_name="annotate_config", version_base=None)
def main(cfg: AnnotateConfig):
    """Do zero shot classification on specified classes"""
    # load precomputed embeddings
    class_emb = torch.load(cfg.class_embedding_pt_path, weights_only=True)

    slide_emb = torch.load(cfg.features_pt_path, weights_only=True)  # N_tiles 512

    # zero-shot classification
    cos_sim = torch.nn.functional.cosine_similarity(
        slide_emb.unsqueeze(1), class_emb.unsqueeze(0), dim=2
    )  # N_tiles N_classes
    df = pd.DataFrame(cos_sim.numpy(), columns=cfg.classes)
    logger.info(df)
    df.to_csv(cfg.output_csv_path, index=False)

    df["class"] = df.idxmax(axis=1)
    logger.info(df["class"].value_counts())

    if cfg.interrogate:
        interrogate_function(
            cfg.slide_path, cfg.patch_path, cfg.interrogation_report_path, df
        )


if __name__ == "__main__":
    main()
