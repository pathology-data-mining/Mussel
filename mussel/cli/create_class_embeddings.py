import open_clip
from dataclasses import dataclass
from typing import List, Optional
import torch
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
import hydra

@dataclass
class ClassEmbeddingConfig:
    classes: List[str] = MISSING
    output_pt_path: str = MISSING
    quiltnet_model_path: str = 'hf-hub:wisdomik/QuiltNet-B-16-PMB'

cs = ConfigStore.instance()
cs.store(name="class_embedding_config", node=ClassEmbeddingConfig)

@hydra.main(config_path=".", config_name="class_embedding_config", version_base=None)
def main(cfg: ClassEmbeddingConfig):
    model, _, _ = open_clip.create_model_and_transforms(cfg.quiltnet_model_path)
    tokenizer = open_clip.get_tokenizer(cfg.quiltnet_model_path)
    embs = []
    for idx, class_text in enumerate(cfg.classes):
        text = tokenizer(class_text)
        with torch.no_grad():
            text_features = model.encode_text(text)
        embs.append((idx, text_features))
    embs.sort(key=lambda x: int(x[0]))
    embs = [x[1] for x in embs]
    class_emb = torch.stack(embs).squeeze(1)
    torch.save(class_emb, cfg.output_pt_path)

if __name__ == "__main__":
    main()
