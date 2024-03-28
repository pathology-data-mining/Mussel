import base64
import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import h5py
import hydra
import openslide
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from loguru import logger


@dataclass
class AnnotateConfig:
    features_pt_path: str
    class_json_path: str
    output_csv_path: str
    interrogate: bool = False
    slide_path: Optional[str] = None
    patch_path: Optional[str] = None
    interrogation_report_path: Optional[str] = None


def load_classes(class_json_path):
    with open(class_json_path, "r") as f:
        class_dict = json.load(f)
        # change keys to int
        class_dict = {int(k): v for k, v in class_dict.items()}
    return class_dict


def load_class_embs(class_json_path, class_dict):
    class_emb_path = class_json_path.replace(".json", ".pt")
    if os.path.exists(class_emb_path):
        class_emb = torch.load(class_emb_path)
    else:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-16-PMB')
        tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-16-PMB')
        embs = []
        for class_id, class_text in class_dict.items():
            text = tokenizer(class_text)
            with torch.no_grad():
                text_features = model.encode_text(text)
            embs.append((class_id, text_features))
        embs.sort(key=lambda x: int(x[0]))
        embs = [x[1] for x in embs]
        class_emb = torch.stack(embs).squeeze(1)
        torch.save(class_emb, class_emb_path)
    return class_emb


def interrogate_function(slide_path, patch_path, interrogation_report_path, df):
    slide = openslide.OpenSlide(slide_path)
    
    with h5py.File(patch_path, "r") as f:
        patch_size = f["coords"].attrs["patch_size"]
        patch_level = f["coords"].attrs["patch_level"]
        logger.info(len(f['coords']))
        assert len(f["coords"]) == len(df), logger.info(f"{len(f['coords'])} vs {len(df)} tiles, aborting")
        coords = f['coords'][:]
    
    df['tile_index'] = df.index

    html_content = ""
    for class_name, sub_df in df.groupby("class"):
        html_content += f"<h2>{class_name}</h2>"
        for _, row in sub_df.iterrows():
            tile_index = row['tile_index']
            tile_coords = coords[tile_index]
            tile = slide.read_region(tile_coords, patch_level, (patch_size, patch_size))
            tile = tile.resize((200, 200))
            tile_bytes = BytesIO()
            tile.save(tile_bytes, format='PNG')
            tile_base64 = base64.b64encode(tile_bytes.getvalue()).decode('utf-8')
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

cs = ConfigStore.instance()
cs.store(name="annotate_config", node=AnnotateConfig)

@hydra.main(config_path=".", config_name="annotate_config", version_base=None)
def main(cfg: AnnotateConfig):
    """Do zero shot classification on specified classes

    Keyword arguments:


    """
    # load precomputed embeddings
    class_dict = load_classes(cfg.class_json_path)
    class_emb = load_class_embs(cfg.class_json_path, class_dict)  # N_classes 512

    slide_emb = torch.load(cfg.features_pt_path)  # N_tiles 512

    # zero-shot classification
    cos_sim = torch.nn.functional.cosine_similarity(slide_emb.unsqueeze(1), class_emb.unsqueeze(0), dim=2)  # N_tiles N_classes
    df = pd.DataFrame(cos_sim.numpy(), columns=[class_dict[i] for i in range(len(class_dict))])
    logger.info(df)
    df.to_csv(cfg.output_csv_path, index=False)

    df['class'] = df.idxmax(axis=1)
    logger.info(df['class'].value_counts())

    if cfg.interrogate:
        interrogate_function(cfg.slide_path, cfg.patch_path, cfg.interrogation_report_path, df)


if __name__ == "__main__":
    main()
