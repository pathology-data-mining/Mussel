"""Regression tests for models whose embeddings were silently wrong.

No weights are needed: each test checks the transform a model hands the
dataset, the tokens it pools, or its forward pass on tiny / synthetic modules.
Modules are put in inference mode with ``train(False)``.
"""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms

import mussel.models  # noqa: F401  # triggers all @register_model decorators
from mussel.models.base import IMAGENET_MEAN, IMAGENET_STD
from mussel.models.clip import ClipModel
from mussel.models.conch import Conch15Model
from mussel.models.conch_v1 import ConchV1Model, _ConchVisualEncoder
from mussel.models.gigapath import GigapathModel
from mussel.models.googlepath import GooglePathModel
from mussel.models.kaiko import (KaikoViTB8Model, KaikoViTB16Model,
                                 KaikoViTL14Model, KaikoViTS8Model,
                                 KaikoViTS16Model)
from mussel.models.lunit import LunitViTS8Model, LunitViTS16Model
from mussel.models.madeleine import MadeleineSlideEncoderModel, _MadeleineModel
from mussel.models.transpath import TransPathModel, _ConvStem
from mussel.models.virchow import Virchow2Model, VirchowModel

_KAIKO_CLASSES = [
    KaikoViTS8Model,
    KaikoViTS16Model,
    KaikoViTB8Model,
    KaikoViTB16Model,
    KaikoViTL14Model,
]


def _make_model(cls, obj, device=None, use_gpu=False):
    """Create a model instance without calling __init__."""
    instance = cls.__new__(cls)
    instance.obj = obj
    instance.device = device or torch.device("cpu")
    instance.use_gpu = use_gpu
    return instance


def _rgb_image(size, seed=0):
    """Random RGB PIL image; ``size`` is an int or a (width, height) tuple."""
    w, h = (size, size) if isinstance(size, int) else size
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def _types(compose):
    return [type(t) for t in compose.transforms]


# ---------------------------------------------------------------------------
# Virchow2: register tokens must not be averaged with the patch tokens
# ---------------------------------------------------------------------------


class _FixedTokens(nn.Module):
    """Returns a fixed token tensor; optionally reports ``num_reg_tokens``."""

    def __init__(self, tokens, num_reg_tokens=None):
        super().__init__()
        self.tokens = tokens
        if num_reg_tokens is not None:
            self.num_reg_tokens = num_reg_tokens

    def forward(self, x):
        return self.tokens


def _tokens(n_reg, n_patches=16, dim=8):
    cls_tok = torch.randn(1, 1, dim)
    reg = torch.full((1, n_reg, dim), 1000.0)  # would dominate a wrong mean
    patches = torch.randn(1, n_patches, dim)
    return torch.cat([cls_tok, reg, patches], dim=1), cls_tok[:, 0], patches


class TestVirchowPatchTokens:
    def test_virchow2_skips_four_register_tokens(self):
        tokens, cls_tok, patches = _tokens(n_reg=4)
        m = _make_model(Virchow2Model, MagicMock(return_value=tokens))
        out = m.get_model_fun()(torch.rand(1, 3, 224, 224))
        expected = torch.cat([cls_tok, patches.mean(dim=1)], dim=1)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    def test_virchow_v1_keeps_tokens_from_index_1(self):
        tokens, cls_tok, patches = _tokens(n_reg=0)
        m = _make_model(VirchowModel, MagicMock(return_value=tokens))
        out = m.get_model_fun()(torch.rand(1, 3, 224, 224))
        expected = torch.cat([cls_tok, patches.mean(dim=1)], dim=1)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    def test_register_count_read_from_model(self):
        tokens, cls_tok, patches = _tokens(n_reg=2)
        m = _make_model(Virchow2Model, _FixedTokens(tokens, num_reg_tokens=2))
        out = m.get_model_fun()(torch.rand(1, 3, 224, 224))
        expected = torch.cat([cls_tok, patches.mean(dim=1)], dim=1)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    def test_real_timm_vit_with_registers(self):
        """A timm ViT with reg_tokens=4 (as Virchow2) pools only its patch tokens."""
        vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            img_size=32,
            reg_tokens=4,
            num_classes=0,
            global_pool="",
        ).train(False)
        x = torch.rand(2, 3, 32, 32)
        with torch.no_grad():
            tokens = vit(x)
        assert tokens.shape[1] == 1 + 4 + 4  # CLS + 4 registers + 2x2 patches
        expected = torch.cat([tokens[:, 0], tokens[:, 5:].mean(dim=1)], dim=1)
        m = _make_model(Virchow2Model, vit)
        out = m.get_model_fun()(x)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# GooglePath: the model must receive values in [0, 1]
# ---------------------------------------------------------------------------


def _fake_tensorflow():
    """Minimal numpy stand-in for the tf calls GooglePath's model_fun makes."""
    tf = types.ModuleType("tensorflow")
    tf.float32 = np.float32
    tf.cast = lambda x, dtype: np.asarray(x, dtype=dtype)
    tf.transpose = lambda x, perm: np.transpose(x, perm)
    tf.image = types.SimpleNamespace(
        resize=lambda t, size, method: t,
        ResizeMethod=types.SimpleNamespace(BICUBIC="bicubic"),
    )
    return tf


class TestGooglePathInputRange:
    def test_preprocessing_has_no_normalize(self):
        m = _make_model(GooglePathModel, MagicMock())
        prep = m.get_preprocessing_fun()
        assert prep is not None
        assert transforms.Normalize not in _types(prep)
        x = prep(_rgb_image(256))
        assert x.shape == (3, 224, 224)
        assert x.min() >= 0.0 and x.max() <= 1.0

    def test_model_receives_unit_range(self):
        captured = {}

        def fake_model(t):
            captured["x"] = t
            return t

        m = _make_model(GooglePathModel, fake_model)
        prep = m.get_preprocessing_fun()
        white = Image.new("RGB", (224, 224), (255, 255, 255))
        batch = torch.stack([prep(white), prep(_rgb_image(224))])
        with patch.dict(sys.modules, {"tensorflow": _fake_tensorflow()}):
            m.get_model_fun()(batch)
        x = captured["x"]
        assert x.shape == (2, 224, 224, 3)  # NHWC
        assert x.min() >= 0.0
        np.testing.assert_allclose(x[0], 1.0)  # white stays 1.0, not 1/255


# ---------------------------------------------------------------------------
# CTransPath: tiles must reach the 224 px Swin at 224 px, ImageNet-normalised
# ---------------------------------------------------------------------------


class TestCTransPathPreprocessing:
    def test_preprocessing_resizes_to_224_with_imagenet_norm(self):
        m = _make_model(TransPathModel, MagicMock())
        prep = m.get_preprocessing_fun()
        assert prep is not None
        assert _types(prep) == [
            transforms.Resize,
            transforms.CenterCrop,
            transforms.ToTensor,
            transforms.Normalize,
        ]
        norm = prep.transforms[-1]
        assert list(norm.mean) == IMAGENET_MEAN and list(norm.std) == IMAGENET_STD
        assert prep(_rgb_image(256)).shape == (3, 224, 224)
        assert prep(_rgb_image((300, 256))).shape == (3, 224, 224)

    def test_256_px_tile_runs_through_swin(self):
        swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            embed_layer=_ConvStem,
            pretrained=False,
            num_classes=0,
        ).train(False)
        m = _make_model(TransPathModel, swin)
        prep = m.get_preprocessing_fun()
        x = prep(_rgb_image(256)).unsqueeze(0)
        out = m.get_model_fun()(x)
        assert out.shape == (1, 768)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Prov-GigaPath: Resize(256, bicubic) + CenterCrop(224), per the model card
# ---------------------------------------------------------------------------


def test_gigapath_preprocessing_matches_card():
    m = _make_model(GigapathModel, MagicMock())
    prep = m.get_preprocessing_fun()
    assert _types(prep) == [
        transforms.Resize,
        transforms.CenterCrop,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    resize, crop = prep.transforms[0], prep.transforms[1]
    assert resize.size == 256
    assert resize.interpolation == transforms.InterpolationMode.BICUBIC
    assert crop.size == (224, 224)
    assert prep(_rgb_image(256)).shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# Kaiko: model-card transform, ViT-L/14 at 224 px via dynamic_img_size
# ---------------------------------------------------------------------------


class TestKaiko:
    @pytest.mark.parametrize("cls", _KAIKO_CLASSES, ids=lambda c: c.__name__)
    def test_preprocessing_matches_card(self, cls):
        m = _make_model(cls, MagicMock())
        prep = m.get_preprocessing_fun()
        assert _types(prep) == [
            transforms.Resize,
            transforms.CenterCrop,
            transforms.ToTensor,
            transforms.Normalize,
        ]
        assert prep.transforms[0].size == 224
        assert prep.transforms[1].size == (224, 224)
        norm = prep.transforms[-1]
        assert tuple(norm.mean) == (0.5, 0.5, 0.5)
        assert tuple(norm.std) == (0.5, 0.5, 0.5)
        assert prep(_rgb_image(256)).shape == (3, 224, 224)

    @pytest.mark.parametrize(
        "cls,expected_kwargs",
        [
            (KaikoViTL14Model, {"dynamic_img_size": True}),
            (KaikoViTS8Model, {}),
            (LunitViTS8Model, {}),
            (LunitViTS16Model, {}),
        ],
        ids=["KaikoViTL14", "KaikoViTS8", "LunitViTS8", "LunitViTS16"],
    )
    def test_create_model_kwargs(self, cls, expected_kwargs):
        with patch(
            "mussel.models.base.timm.create_model", return_value=nn.Linear(2, 2)
        ) as create:
            cls(None, use_gpu=False)
        create.assert_called_once_with(
            cls._default_model_type.path,
            pretrained=True,
            num_classes=0,
            **expected_kwargs,
        )


# ---------------------------------------------------------------------------
# CONCH v1.0: ln_contrast after attention pooling; bicubic resize
# ---------------------------------------------------------------------------


def _tiny_vit(*args, **kwargs):
    """Stand-in for the ViT-B/16 trunk: 768-dim tokens, one block, 32 px."""
    return VisionTransformer(
        img_size=32, patch_size=16, embed_dim=768, depth=1, num_heads=12, num_classes=0
    )


class TestConchV1:
    def _checkpoint(self, tmp_path):
        with patch("mussel.models.conch_v1.timm.create_model", _tiny_vit):
            ref = _ConchVisualEncoder()
        ckpt = {f"visual.trunk.{k}": v for k, v in ref.trunk.state_dict().items()}
        ckpt.update(
            {
                f"visual.attn_pool_contrast.{k}": v
                for k, v in ref.attn_pool_contrast.state_dict().items()
            }
        )
        torch.manual_seed(0)
        ln_weight = torch.rand(512) + 0.5
        ln_bias = torch.randn(512)
        ckpt["visual.ln_contrast.weight"] = ln_weight
        ckpt["visual.ln_contrast.bias"] = ln_bias
        # Unused head present in the real checkpoint.
        ckpt["visual.proj_contrast"] = torch.randn(512, 512)
        path = tmp_path / "pytorch_model.bin"
        torch.save(ckpt, path)
        return str(path), ln_weight, ln_bias

    def test_ln_contrast_loaded_and_applied(self, tmp_path):
        path, ln_weight, ln_bias = self._checkpoint(tmp_path)
        with patch("mussel.models.conch_v1.timm.create_model", _tiny_vit):
            model = ConchV1Model._load(path)
        torch.testing.assert_close(model.ln_contrast.weight.data, ln_weight)
        torch.testing.assert_close(model.ln_contrast.bias.data, ln_bias)

        x = torch.rand(2, 3, 32, 32)
        with torch.no_grad():
            pooled = model.attn_pool_contrast(model.trunk.forward_features(x))
            expected = F.layer_norm(pooled, (512,), ln_weight, ln_bias)
            out = model(x)
        assert out.shape == (2, 512)
        torch.testing.assert_close(out, expected)

    def test_preprocessing_bicubic(self):
        m = _make_model(ConchV1Model, MagicMock())
        prep = m.get_preprocessing_fun()
        resize = prep.transforms[0]
        assert isinstance(resize, transforms.Resize)
        assert resize.size == 448
        assert resize.interpolation == transforms.InterpolationMode.BICUBIC


# ---------------------------------------------------------------------------
# MADELEINE: faithful port of upstream ABMILEmbedder + encode_he
# ---------------------------------------------------------------------------


class _UpstreamBatchedABMIL(nn.Module):
    """madeleine/models/abmil.py BatchedABMIL (dropout=True, softmax)."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(0.25)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(0.25)
        )
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.attention_c(self.attention_a(x).mul(self.attention_b(x)))
        return F.softmax(A, dim=1)


class _UpstreamMadeleine(nn.Module):
    """madeleine/models/Model.py MADELEINE.encode_he with ABMILEmbedder.

    Parameter names mirror upstream so the same state dict loads into both.
    """

    def __init__(self, in_dim, hidden, n_heads):
        super().__init__()
        self.n_heads = n_heads
        pre_attn = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden * n_heads),
            nn.LayerNorm(hidden * n_heads),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        attn = nn.ModuleList(
            [_UpstreamBatchedABMIL(hidden, 512) for _ in range(n_heads)]
        )
        self.wsi_embedders = nn.ModuleDict({"pre_attn": pre_attn, "attn": attn})
        self.token_projector = nn.Linear(512 * n_heads, 128)
        self.projector = nn.Linear(512 * n_heads, 512)

    def forward(self, feats):
        einops = pytest.importorskip("einops")
        bs = feats.shape[0]
        emb = self.wsi_embedders["pre_attn"](feats)
        emb = einops.rearrange(emb, "b t (e c) -> b t e c", c=self.n_heads)
        attention = torch.stack(
            [net(emb[:, :, :, i]) for i, net in enumerate(self.wsi_embedders["attn"])],
            dim=-1,
        )
        slide = torch.sum(emb * attention, dim=1)
        d_out, n_heads = slide.shape[-2], slide.shape[-1]
        slide = slide.view(bs, d_out * n_heads)
        return self.projector(slide)


_MADELEINE_CFG = {
    "patch_embedding_dim": 512,
    "wsi_encoder_hidden_dim": 512,
    "n_heads": 4,
    "activation": "softmax",
}


def _write_madeleine_dir(path, state_dict):
    path.mkdir(parents=True, exist_ok=True)
    # The released checkpoint is DDP-wrapped.
    torch.save({f"module.{k}": v for k, v in state_dict.items()}, path / "model.pt")
    (path / "model_config.json").write_text(json.dumps(_MADELEINE_CFG))


class TestMadeleine:
    def _reference(self):
        torch.manual_seed(0)
        ref = _UpstreamMadeleine(512, 512, 4).train(False)
        # Non-trivial LayerNorm affine params so a wrong head split shows up.
        for mod in ref.modules():
            if isinstance(mod, nn.LayerNorm):
                nn.init.normal_(mod.weight, 1.0, 0.5)
                nn.init.normal_(mod.bias, 0.0, 0.5)
        return ref

    def test_forward_matches_upstream(self):
        """Same weights, same input: the port's output equals upstream encode_he."""
        ref = self._reference()
        model = _MadeleineModel(512, 512, 4).train(False)
        model.load_state_dict(ref.state_dict(), strict=True)
        feats = torch.randn(2, 30, 512)
        with torch.no_grad():
            out, expected = model(feats), ref(feats)
        assert out.shape == (2, 512)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)

    def test_matches_upstream_encode_he(self, tmp_path):
        ref = self._reference()
        _write_madeleine_dir(tmp_path / "madeleine", ref.state_dict())
        with patch(
            "mussel.models.madeleine.hf_hub_download",
            side_effect=AssertionError("local dir must not download"),
        ):
            m = MadeleineSlideEncoderModel(str(tmp_path / "madeleine"), use_gpu=False)
        feats = torch.randn(1, 50, 512)
        with torch.no_grad():
            expected = ref(feats)[0]
        out = m.get_model_fun()(feats)
        assert out.shape == (512,)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)

    def test_save_roundtrip_loads_from_directory(self, tmp_path):
        ref = self._reference()
        _write_madeleine_dir(tmp_path / "src", ref.state_dict())
        m = MadeleineSlideEncoderModel(str(tmp_path / "src"), use_gpu=False)
        m.save(str(tmp_path / "saved"))
        m2 = MadeleineSlideEncoderModel(str(tmp_path / "saved"), use_gpu=False)
        feats = torch.randn(1, 20, 512)
        torch.testing.assert_close(m2.get_model_fun()(feats), m.get_model_fun()(feats))


# ---------------------------------------------------------------------------
# CLIP loaded from a saved model file must get a preprocessing transform
# ---------------------------------------------------------------------------


class TestClipFromFile:
    def _saved_model(self, tmp_path, visual_attrs):
        model = nn.Linear(2, 2)
        model.visual = nn.Module()
        for k, v in visual_attrs.items():
            setattr(model.visual, k, v)
        path = tmp_path / "clip_model.bin"
        _make_model(ClipModel, model).save(str(path))
        return str(path)

    def _assert_openai_eval_transform(self, prep, size=224):
        resize = next(t for t in prep.transforms if isinstance(t, transforms.Resize))
        crop = next(t for t in prep.transforms if isinstance(t, transforms.CenterCrop))
        norm = prep.transforms[-1]
        assert resize.size == size
        assert resize.interpolation == transforms.InterpolationMode.BICUBIC
        assert crop.size == (size, size)
        assert tuple(norm.mean) == tuple(OPENAI_DATASET_MEAN)
        assert tuple(norm.std) == tuple(OPENAI_DATASET_STD)
        assert prep(_rgb_image(256)).shape == (3, size, size)

    def test_uses_preprocess_cfg(self, tmp_path):
        cfg = {
            "size": (224, 224),
            "mode": "RGB",
            "mean": OPENAI_DATASET_MEAN,
            "std": OPENAI_DATASET_STD,
            "interpolation": "bicubic",
            "resize_mode": "shortest",
            "fill_color": 0,
        }
        path = self._saved_model(
            tmp_path, {"preprocess_cfg": cfg, "image_size": (224, 224)}
        )
        m = ClipModel(path, use_gpu=False)
        self._assert_openai_eval_transform(m.get_preprocessing_fun())

    def test_falls_back_without_preprocess_cfg(self, tmp_path):
        path = self._saved_model(tmp_path, {})
        m = ClipModel(path, use_gpu=False)
        self._assert_openai_eval_transform(m.get_preprocessing_fun())


# ---------------------------------------------------------------------------
# CONCH v1.5: CenterCrop(448) for TITAN parity, a no-op on square tiles
# ---------------------------------------------------------------------------


class TestConch15CenterCrop:
    def test_has_center_crop_and_keeps_bilinear(self):
        m = _make_model(Conch15Model, MagicMock())
        prep = m.get_preprocessing_fun()
        assert _types(prep) == [
            transforms.Resize,
            transforms.CenterCrop,
            transforms.ToTensor,
            transforms.Normalize,
        ]
        assert prep.transforms[0].interpolation == (
            transforms.InterpolationMode.BILINEAR
        )
        assert prep.transforms[1].size == (448, 448)

    @pytest.mark.parametrize("size", [448, 512])
    def test_center_crop_is_noop_on_square_tiles(self, size):
        old = transforms.Compose(
            [
                transforms.Resize(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        new = _make_model(Conch15Model, MagicMock()).get_preprocessing_fun()
        img = _rgb_image(size, seed=size)
        assert torch.equal(new(img), old(img))
