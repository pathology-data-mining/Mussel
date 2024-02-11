from argparse import ArgumentParser
import os
import json
import torch
import pandas as pd


def load_classes(class_json_path):
    with open(class_json_path, "r") as f:
        class_dict = json.load(f)
        # change keys to int
        class_dict = {int(k): v for k, v in class_dict.items()}
    return class_dict


def load_class_embs(class_json_path):
    class_emb_path = class_json_path.replace(".json", ".pt")
    if os.path.exists(class_emb_path):
        class_emb = torch.load(class_emb_path)
    else:
        from foundational_inference.utils import load_quilt
        quilt = load_quilt()
        embs = []
        for class_id, class_text in CLASS_DICT.items():
            text = quilt["tokenizer"](class_text)
            with torch.no_grad():
                text_features = quilt["model"].encode_text(text)
            embs.append((class_id, text_features))
        embs.sort(key=lambda x: int(x[0]))
        embs = [x[1] for x in embs]
        class_emb = torch.stack(embs).squeeze(1)
        torch.save(class_emb, class_emb_path)
    return class_emb


PARSER = ArgumentParser()
PARSER.add_argument("--slide_emb_path", type=str)
PARSER.add_argument("--class_json_path", type=str)
PARSER.add_argument("--output_path", type=str)
ARGS = PARSER.parse_args()

# load precomputed embeddings
CLASS_DICT = load_classes(ARGS.class_json_path)
CLASS_EMB = load_class_embs(ARGS.class_json_path)  # N_classes 512

SLIDE_EMB = torch.load(ARGS.slide_emb_path)  # N_tiles 512

# zero-shot classification
COS_SIM = torch.nn.functional.cosine_similarity(SLIDE_EMB.unsqueeze(1), CLASS_EMB.unsqueeze(0), dim=2)  # N_tiles N_classes
DF = pd.DataFrame(COS_SIM.numpy(), columns=[CLASS_DICT[i] for i in range(len(CLASS_DICT))])
print(DF)
DF.to_csv(ARGS.output_path, index=False)

DF['class'] = DF.idxmax(axis=1)
print(DF['class'].value_counts())
