from argparse import ArgumentParser
import os
import json
import torch
import pandas as pd
from io import BytesIO
import base64
import h5py
import openslide


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


def interrogate_function(svs_path, patch_path, interrogation_report_path, df):
    slide = openslide.OpenSlide(svs_path)
    
    with h5py.File(patch_path, "r") as f:
        patch_size = f["coords"].attrs["patch_size"]
        patch_level = f["coords"].attrs["patch_level"]
        print(len(f['coords']))
        assert len(f["coords"]) == len(df), print(f"{len(f['coords'])} vs {len(df)} tiles, aborting")
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


def main(slide_emb_path,
         class_json_path,
         output_path,
         interrogate=False,
         svs_path=None,
         patch_path=None,
         interrogation_report_path=None):
    """Do zero shot classification on specified classes

    Keyword arguments:


    """
    # load precomputed embeddings
    CLASS_DICT = load_classes(class_json_path)
    CLASS_EMB = load_class_embs(class_json_path, CLASS_DICT)  # N_classes 512

    SLIDE_EMB = torch.load(slide_emb_path)  # N_tiles 512

    # zero-shot classification
    COS_SIM = torch.nn.functional.cosine_similarity(SLIDE_EMB.unsqueeze(1), CLASS_EMB.unsqueeze(0), dim=2)  # N_tiles N_classes
    DF = pd.DataFrame(COS_SIM.numpy(), columns=[CLASS_DICT[i] for i in range(len(CLASS_DICT))])
    print(DF)
    DF.to_csv(output_path, index=False)

    DF['class'] = DF.idxmax(axis=1)
    print(DF['class'].value_counts())

    if interrogate:
        interrogate_function(svs_path, patch_path, interrogation_report_path, DF)


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("--slide_emb_path", type=str)
    PARSER.add_argument("--class_json_path", type=str)
    PARSER.add_argument("--output_path", type=str)
    PARSER.add_argument("--interrogate", action="store_true", help='if true, will prepare zsl report')
    PARSER.add_argument("--svs_path", type=str, help='only used for interrogation')
    PARSER.add_argument('--patch_path', type=str, help='only used for interrogation')
    PARSER.add_argument('--interrogation_report_path', type=str, help='only used for interrogation')
    ARGS = PARSER.parse_args()
    ARGS = vars(ARGS)
    main(**ARGS)
