"""KEEP text tower: loads the checkpoint's ``text.*`` BERT and returns its normalised pooled output."""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import BertConfig, BertModel, BertTokenizer

from mussel.models.keep import KeepTextEncoder

_TEXT_CONFIG = dict(
    vocab_size=40,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    max_position_embeddings=300,
    type_vocab_size=2,
)
_WORDS = [
    "tumor",
    "epithelium",
    "lymphocytes",
    "adipose",
    "tissue",
    "necrosis",
    "stroma",
    "of",
    "the",
    "is",
]


def _keep_dir(tmp_path, extra_text_key=False):
    """A tiny KEEP checkpoint: random BERT under ``text.``, a stray vision key, a vocab."""
    torch.manual_seed(0)
    bert = BertModel(BertConfig(**_TEXT_CONFIG))
    bert.train(False)
    sd = {f"text.{k}": v.contiguous() for k, v in bert.state_dict().items()}
    sd["visual.cls_token"] = torch.zeros(1, 1, 8)  # ignored by the text tower
    if extra_text_key:
        sd["text.unexpected.weight"] = torch.zeros(2)
    save_file(sd, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"projection_dim": 32, "text_config": _TEXT_CONFIG})
    )
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + _WORDS
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    return tmp_path, bert


def test_matches_bert_pooled_output(tmp_path):
    path, bert = _keep_dir(tmp_path)
    texts = [
        "Tumor epithelium",
        "lymphocytes",
        "the stroma of the tumor is necrosis",
        "adipose tissue",
    ]
    got = KeepTextEncoder(str(path)).encode(texts, batch_size=3)

    tok = BertTokenizer(str(path / "vocab.txt"), do_lower_case=True)
    with torch.no_grad():
        ref = bert(
            **tok(
                texts,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        ).pooler_output
    ref = torch.nn.functional.normalize(ref, dim=-1)
    assert got.shape == (4, 32)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got.norm(dim=-1), torch.ones(4))


def test_batching_does_not_change_embeddings(tmp_path):
    path, _ = _keep_dir(tmp_path)
    enc = KeepTextEncoder(str(path))
    texts = [f"{a} {b}" for a in _WORDS[:4] for b in _WORDS[4:7]]
    torch.testing.assert_close(
        enc.encode(texts, batch_size=5), enc.encode(texts, batch_size=64)
    )


def test_mismatched_checkpoint_is_rejected(tmp_path):
    path, _ = _keep_dir(tmp_path, extra_text_key=True)
    with pytest.raises(RuntimeError, match="do not match"):
        KeepTextEncoder(str(path))
