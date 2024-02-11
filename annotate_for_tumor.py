from argparse import ArgumentParser
import os
import json
import torch


PARSER = ArgumentParser()
PARSER.add_argument("--slide_emb_path", type=str)
PARSER.add_argument("--class_json_path", type=str)
PARSER.add_argument("--output_path", type=str)
ARGS = PARSER.parse_args()

with open(ARGS.class_json_path, "r") as f:
    class_dict = json.load(f)

CLASS_EMB_PATH = ARGS.class_json_path.replace(".json", ".pt")
if os.path.exists(CLASS_EMB_PATH):
    CLASS_EMB = torch.load(CLASS_EMB_PATH)
else:
    from foundational_inference.utils import load_quilt
    quilt = load_quilt()
    embs = []
    for class_id, class_text in class_dict.items():
        text = quilt["tokenizer"](class_text)
        with torch.no_grad():
            text_features = quilt["model"].encode_text(text)
        embs.append((class_id, text_features))
    embs.sort(key=lambda x: int(x[0]))
    print(embs)
    embs = [x[1] for x in embs]
    CLASS_EMB = torch.stack(embs).squeeze(1)
    print(CLASS_EMB.shape)
    torch.save(CLASS_EMB, CLASS_EMB_PATH)
