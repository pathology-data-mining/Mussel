"""CONCH v1.5 patch encoder and TITAN slide encoder from MahmoodLab."""

import contextlib
import logging
import math
import types
from pathlib import Path
from typing import Callable, List

import torch
from torchvision import transforms
from transformers import AutoModel

from mussel.models.base import (IMAGENET_MEAN, IMAGENET_STD, TorchModel,
                                get_best_attn_implementation)

try:
    from mussel.utils.model_cache import model_download_lock
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def model_download_lock(model_name, **kwargs):
        yield True


from mussel.models.model_factory import ModelType, register_model

logger = logging.getLogger(__name__)


@register_model(ModelType.CONCH1_5)
class Conch15Model(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize CONCH v1.5 model.

        Args:
            model_path: Path to model file or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.CONCH1_5.path
        model_obj = None
        if not Path(model_path).is_file():
            attn_impl = get_best_attn_implementation()
            # Use locking when downloading from HuggingFace
            with model_download_lock(model_path) as should_download:
                try:
                    titan = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        attn_implementation=attn_impl,
                    )
                except (ValueError, NotImplementedError) as e:
                    # TITAN model doesn't support Flash Attention 2.0 or SDPA yet, fallback to eager
                    if "does not support an attention implementation" in str(
                        e
                    ) or "does not support Flash Attention" in str(e):
                        logger.info(
                            "TITAN model doesn't support optimized attention, using eager mode"
                        )
                        titan = AutoModel.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            attn_implementation="eager",
                        )
                    else:
                        raise
            model_obj, _ = titan.return_conch()
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing transforms for CONCH v1.5.

        Returns:
            Composed transforms for CONCH v1.5 input preprocessing.
        """
        return transforms.Compose(
            [
                transforms.Resize(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def save(self, save_path: str):
        """CONCH1_5 model saving is disabled.

        The CONCH model is extracted from TITAN and contains dynamic modules
        that cannot be reliably pickled/unpickled. Use TITAN_SLIDE directory instead.

        Args:
            save_path: Path where model would be saved (ignored).

        Raises:
            NotImplementedError: Always raised to prevent saving.
        """
        raise NotImplementedError(
            "Saving CONCH1_5 model is not supported. "
            "CONCH is extracted from TITAN model and contains dynamic modules that "
            "cannot be reliably pickled. To cache models, save the TITAN_SLIDE directory instead."
        )


# ---------------------------------------------------------------------------
# TITAN monkey-patch helpers
# ---------------------------------------------------------------------------

def _get_slopes(n: int) -> list:
    """ALiBi attention slopes for ``n`` heads (from TITAN/vision_transformer.py)."""
    if math.log2(n) == int(math.log2(n)):
        p = 2 ** (-2 ** -(math.log2(n) - 3))
        return [p * (p ** i) for i in range(n)]
    nearest = 2 ** math.floor(math.log2(n))
    base = _get_slopes(nearest)
    if nearest == n:
        return base
    extra = _get_slopes(2 * nearest)[0::2][:n - nearest]
    return base + extra

def _titan_get_alibi_gpu_float16(self, w: int, h: int, bg_mask=None):
    """GPU float16 replacement for VisionTransformer.get_alibi().

    The original implementation creates O(N²) numpy float64 arrays on CPU
    (17 GB for N=33k), causing SLURM OOM on large IMPACT resection specimens.
    This version uses torch.cdist in float16 on the model's GPU, reducing peak
    memory from ~82 GB CPU → ~26 GB GPU for N=33k.

    torch.cdist is fused in CUDA and does not create the intermediate (N, N, 2)
    array that numpy broadcasting would require.
    """
    device = next(self.parameters()).device
    dtype = torch.float16

    x_coords = torch.arange(w, device=device, dtype=dtype)
    y_coords = torch.arange(h, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')

    if bg_mask is not None:
        # bg_mask shape is (1, H, W) bool — squeeze to (H, W), index into 2D grid
        mask_2d = bg_mask.to(device).squeeze(0).bool()  # (H, W) or (W*H,) depending on caller
        if mask_2d.dim() == 1:
            # Already flattened (w*h,)
            mask_flat = mask_2d
            pts_x = grid_x.ravel()[mask_flat]
            pts_y = grid_y.ravel()[mask_flat]
        else:
            # 2D mask (W, H) — use as index into grid
            pts_x = grid_x[mask_2d]
            pts_y = grid_y[mask_2d]
    else:
        pts_x = grid_x.ravel()
        pts_y = grid_y.ravel()

    points = torch.stack([pts_x, pts_y], dim=1)  # (N, 2) float16

    # Pairwise Euclidean distances — fused CUDA, no (N, N, 2) intermediate
    dists = torch.cdist(points.float(), points.float(), p=2).to(dtype)  # (N, N)

    slopes = torch.tensor(
        _get_slopes(self.num_heads), dtype=dtype, device=device
    ).view(self.num_heads, 1, 1)

    n_patches = dists.shape[0]
    bias_matrix = -dists.unsqueeze(0) * slopes          # (H, N, N)
    embed_len = n_patches + 1
    all_bias = torch.zeros(
        1, self.num_heads, embed_len, embed_len, dtype=dtype, device=device
    )
    all_bias[:, :, 1:, 1:] = bias_matrix
    return all_bias


def _titan_forward_features_efficient(self, x, coords=None, mask=None, bg_mask=None):
    """Memory-efficient replacement for VisionTransformer.forward_features().

    The original uses `attn_bias.repeat(B, 1, 1, 1)` which creates a full copy
    of the (1, H, N, N) bias tensor — 22 GB for N=30k on A100. This replacement
    uses `expand()` (a zero-copy view) and avoids redundant dtype/device casts
    when the bias is already in the correct format (float16 on GPU from the
    get_alibi monkey-patch).
    """
    B, nc, w, h = x.shape
    x = x.flatten(2, 3).transpose(1, 2)

    if self.pos_encode_type == 'alibi':
        if w * h == 36 and B != 1:
            if not self.local_alibi_status:
                self.prepare_tensor(x, 'local', 'alibi')
            attn_bias = self.local_alibi
        elif w * h == 196 and B != 1:
            if not self.global_alibi_status:
                self.prepare_tensor(x, 'global', 'alibi')
            attn_bias = self.global_alibi
        else:
            # Calls our monkey-patched get_alibi (returns float16 GPU tensor)
            attn_bias = self.get_alibi(w, h, bg_mask) if B == 1 else self.get_alibi(w, h)
            # Use expand instead of repeat: zero-copy view for the B dimension
            attn_bias = attn_bias.expand(x.shape[0], -1, -1, -1)
            # Only cast if dtype/device differ (avoids copy when already float16 on GPU)
            if attn_bias.dtype != x.dtype or attn_bias.device != x.device:
                attn_bias = attn_bias.to(dtype=x.dtype, device=x.device)
    else:
        attn_bias = None

    if self.masked_im_modeling:
        assert mask is not None
        x = self.patch_embed(x)
        x = self.mask_model(x, mask)
    else:
        x = self.patch_embed(x)

    x = self._pos_embed(x, coords, w, h)
    x = self.norm_pre(x)

    # Mask background tokens when evaluating (B=1)
    if bg_mask is not None and B == 1:
        bg_mask_cat = torch.cat(
            (torch.ones((1, 1), dtype=torch.bool, device=x.device), bg_mask.view(1, -1)),
            dim=1,
        )
        x = x[bg_mask_cat].unsqueeze(0)

    if self.grad_checkpointing and not torch.jit.is_scripting():
        from timm.models._manipulate import checkpoint_seq
        x = checkpoint_seq(self.blocks, x, attn_bias, bg_mask)
    else:
        x = self.blocks(x, attn_bias, bg_mask)

    x = self.norm(x)
    return x


@register_model(ModelType.TITAN_SLIDE)
class TitanSlideEncoderModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        """Initialize TITAN slide encoder model.

        This is the slide-level encoder from MahmoodLab/TITAN that aggregates
        CONCH patch-level features into slide-level representations.

        Args:
            model_path: Path to slide encoder model directory or HuggingFace repo ID.
            use_gpu: Whether to use GPU (default: True).
            gpu_device_id: GPU device ID or list of IDs for multi-GPU (default: None).
        """
        if model_path is None:
            model_path = ModelType.TITAN_SLIDE.path

        # Validate that model_path is appropriate for TITAN slide encoder
        if Path(model_path).is_file():
            raise ValueError(
                f"TITAN_SLIDE requires a directory path or HuggingFace repo ID, not a file. "
                f"Got: {model_path}\n"
                f"Hint: Use slide_model_type=TITAN_SLIDE (not model_type) and "
                f"slide_model_path='path/to/TITAN_SLIDE/'"
            )

        model_obj = None
        # Load the TITAN model from HuggingFace or saved directory
        # TITAN doesn't support Flash Attention 2.0, so we use eager mode
        # Use locking when downloading from HuggingFace
        with model_download_lock(model_path) as should_download:
            try:
                model_obj = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    attn_implementation="eager",
                )
            except TypeError:
                # Fallback for older transformers that don't support attn_implementation
                model_obj = AutoModel.from_pretrained(
                    model_path, trust_remote_code=True
                )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_model_fun(self) -> Callable:
        """Get model inference function for TITAN slide encoder.

        Applies monkey-patches to the TITAN vision encoder to fix CPU/GPU RAM OOM
        on large IMPACT slides (>25k patches):

        1. ``get_alibi`` → GPU float16 via torch.cdist
           Eliminates ~82 GB CPU RAM peak for N=30k patches.
        2. ``forward_features`` → uses expand() instead of repeat()
           Avoids a 22 GB copy of the bias tensor for N=30k.

        Memory budget on A100 (80 GB): bias (22 GB) + model (2 GB) + QK^T (22 GB)
        + intermediates (~3 GB) ≈ 49 GB → fits with headroom.
        To further reduce allocator fragmentation set
        ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` in the process
        environment *before* Python starts (the CUDA allocator reads this env var
        once at initialization, before any model is loaded).
        """
        # Apply monkey-patches to the vision encoder
        vision_enc = self.obj.vision_encoder
        vision_enc.get_alibi = types.MethodType(_titan_get_alibi_gpu_float16, vision_enc)
        vision_enc.forward_features = types.MethodType(_titan_forward_features_efficient, vision_enc)
        logger.debug(
            "TITAN: applied GPU float16 get_alibi + expand-based forward_features monkey-patches"
        )

        # Use SDPBackend.EFFICIENT_ATTENTION in model_fun to prevent the math kernel
        # from materializing the full QK^T matrix (~22 GB for N=18k), which would OOM.
        # EFFICIENT_ATTENTION requires CUDA compute >= 8.0 (A100+); fall back to the
        # default SDPA kernel selection on older hardware (P40, V100, etc.).
        _efficient_ctx = contextlib.nullcontext
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            if self.device.type == 'cuda':
                major, _ = torch.cuda.get_device_capability(self.device)
                if major >= 8:
                    _efficient_ctx = lambda: sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        except (ImportError, RuntimeError):
            pass

        def model_fun(patch_features, coords, patch_size):
            """Run TITAN slide encoder on patch features with coordinates and patch size."""
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
                _efficient_ctx(),
            ):
                patch_features = patch_features.to(self.device, non_blocking=True)
                coords = coords.to(self.device, non_blocking=True)
                return (
                    self.obj.encode_slide_from_patch_features(
                        patch_features, coords, patch_size
                    )
                    .squeeze()
                    .float()
                    .cpu()
                )

        return model_fun

    def get_preprocessing_fun(self) -> Callable:
        """Get preprocessing function for slide encoder.

        Slide encoders work on patch features, not images, so no preprocessing needed.

        Returns:
            None, as slide encoders don't preprocess images.
        """
        return None

    def save(self, save_path: str):
        """Save TITAN slide encoder model to disk.

        TITAN models loaded from HuggingFace cannot be pickled directly,
        so we save using HuggingFace's save_pretrained method to a directory.

        Args:
            save_path: Path to save the model (must be a directory, not a file).

        Raises:
            ValueError: If save_path has a file extension (like .pth or .pkl).
        """
        # Check if save_path looks like a file (has extension)
        if Path(save_path).suffix:
            raise ValueError(
                f"TITAN_SLIDE model cannot be saved to a file ({save_path}). "
                f"It must be saved as a directory using HuggingFace's format. "
                f"Remove the file extension and try again."
            )

        # Create directory if it doesn't exist
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get the base model (unwrap DataParallel if needed)
        model_to_save = self.obj.module if hasattr(self.obj, "module") else self.obj

        # Save using HuggingFace's save_pretrained method
        try:
            model_to_save.save_pretrained(save_path)
            logger.info(f"Saved TITAN slide encoder to {save_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to save TITAN_SLIDE model to {save_path}: {e}"
            ) from e
