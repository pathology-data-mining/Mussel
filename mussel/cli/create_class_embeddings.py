import json
from dataclasses import dataclass
from typing import List, Optional

import hydra
import open_clip
import torch
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from mussel.models.model_factory import ModelType


@dataclass
class ClassEmbeddingConfig:
    """
    classes (List[str]): List of class names for which to compute embeddings.
    output_pt_path (str): Path to save the computed class embeddings in PyTorch format.
    model_path (Optional[str]): Path to the model weights, if applicable.
    model_type (ModelType): Type of model to use for computing embeddings: CLIP
        (open_clip text tower) or KEEP (BERT text tower; L2-normalised embeddings
        comparable with KEEP tile features, or with features mapped into KEEP's
        space by an adapter, see annotate's adapter_path).
    """

    classes: List[str] = MISSING
    output_pt_path: str = MISSING
    model_path: Optional[str] = None
    model_type: ModelType = ModelType.CLIP


desc_doc = """== ${hydra.help.app_name} ==
Computes class embeddings for zero-shot classification using a specified model.
"""

parameter_doc = f"""
== Available Parameters ==
{ClassEmbeddingConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="class_embedding_config", node=ClassEmbeddingConfig)


@hydra.main(config_path=".", config_name="class_embedding_config", version_base=None)
def main(cfg: ClassEmbeddingConfig):
    """Generate class embeddings for zero-shot tissue classification."""
    if cfg.model_type not in (ModelType.CLIP, ModelType.KEEP):
        raise ValueError(
            f"create_class_embeddings supports model_type CLIP or KEEP, not {cfg.model_type.name}"
        )
    if cfg.model_type == ModelType.KEEP:
        from mussel.models.keep import KeepTextEncoder

        class_emb = KeepTextEncoder(cfg.model_path).encode(list(cfg.classes))
        torch.save(class_emb, cfg.output_pt_path)
        _write_classes_sidecar(cfg)
        return
    if cfg.model_path is None:
        cfg.model_path = cfg.model_type.path
    model, _, _ = open_clip.create_model_and_transforms(cfg.model_path)
    # open_clip returns the model in training mode: without this, dropout makes the same
    # class text embed differently on every run (cosine ~0.987 for QuiltNet).
    model.train(False)
    tokenizer = open_clip.get_tokenizer(cfg.model_path)

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
    _write_classes_sidecar(cfg)


def classes_sidecar_path(class_embedding_pt_path) -> str:
    """``<embeddings>.classes.json``: the class names, in row order, and the text model."""
    return f"{class_embedding_pt_path}.classes.json"


def _write_classes_sidecar(cfg: ClassEmbeddingConfig) -> None:
    # annotate checks its classes list against this, so a reordered list can't mislabel scores.
    with open(classes_sidecar_path(cfg.output_pt_path), "w") as f:
        json.dump({"classes": list(cfg.classes), "model_type": cfg.model_type.name}, f, indent=1)


if __name__ == "__main__":
    main()
