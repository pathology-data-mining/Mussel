# CLAUDE.md

This file provides guidance for Claude Code when working with the Mussel codebase.

## Project Overview

Mussel is a computational pathology toolkit for processing whole-slide images (WSI). It provides CLI tools for tiling, feature extraction, annotation, and aggregation using various foundation models (OpenCLIP, ResNet-50, TransPath, Virchow, Virchow2, Prov-GigaPath, H-Optimus-0, GooglePath, CONCH).

- **Language**: Python 3.10-3.11
- **Package Manager**: `uv`
- **Build System**: setuptools (via `pyproject.toml`)
- **License**: GPL-3.0

## Repository Structure

```
mussel/
  cli/          # CLI entry points (tessellate, extract_features, annotate, etc.)
  models/       # Foundation model implementations and factory
  datasets/     # Data loading (HDF5, tile coords, flat images)
  utils/        # Feature extraction, segmentation, file I/O, ML utilities
tests/
  testdata/     # Test slide images and fixtures
  mussel/       # Mirrors main package structure
presets/        # Preset configurations (biopsy, resection, TCGA)
```

## Common Commands

### Install dependencies
```bash
uv sync --extra torch-gpu        # PyTorch with CUDA
uv sync --extra torch-cpu        # PyTorch CPU-only
uv sync --extra tensorflow-gpu   # TensorFlow with CUDA
```

### Run tests
```bash
uv run pytest tests
```

### Run a specific test file
```bash
uv run pytest tests/mussel/cli/test_tessellate.py
```

### Run a specific test
```bash
uv run pytest tests/mussel/cli/test_tessellate.py::test_function_name
```

## Code Style & Conventions

- **Formatter**: `black`
- **Import sorting**: `isort`
- **Type checking**: `mypy`
- **Logging**: Standard `logging` module (not loguru)
- Type hints are used throughout; follow existing patterns
- Models use the factory pattern (`ModelFactory.create()`)
- Dataset processing uses the strategy pattern (`get_dataset_processor()`)

### Import style

All imports belong at the top of the file. **Do not place imports inside functions or methods** unless one of these specific exceptions applies:

1. **Optional / guarded dependency** — the import is inside a `try/except ImportError` block because the package may not be installed (e.g. `fsspec`, `flash_attn`, `tensorflow`, `gigapath`).
2. **Platform-conditional import** — the import only makes sense on certain OSes (e.g. `fcntl` on Linux, `msvcrt` on Windows).
3. **Circular-import workaround** — moving the import to the top would create a circular dependency.

Everything else — stdlib modules (`os`, `tempfile`, `warnings`, `traceback`, `collections`, `multiprocessing`, `functools`), third-party packages that are always installed (`numpy`, `torch`, `omegaconf`), and local modules — must be at the top.

## Hydra Configuration System

All 13 CLI commands use Hydra with **structured configs only** (no YAML files). The pattern is:

1. Define a `@dataclass` for the command's config with typed fields and defaults
2. Register it with `ConfigStore`
3. Decorate `main()` with `@hydra.main(version_base=None, config_path=".", config_name="...")`

```python
@dataclass
class ExtractFeaturesConfig:
    slide_path: Optional[str] = None
    batch_size: int = 64
    model_type: ModelType = ModelType.CLIP

cs = ConfigStore.instance()
cs.store(name="extract_features_config", node=ExtractFeaturesConfig)

@hydra.main(version_base=None, config_path=".", config_name="extract_features_config")
def main(cfg: ExtractFeaturesConfig):
    ...
```

**Users pass config via Hydra command-line overrides** (not `--flag` style):
```bash
extract_features slide_path=slide.svs model_type=VIRCHOW batch_size=128
tessellate slide_path=slide.svs seg_config=biopsy   # config group preset
```

**Nested config groups** are used for segmentation presets (`seg_config=default|biopsy|resection|tcga`), defined as dataclass inheritance (e.g., `BiopsySegConfig(SegConfig)`).

**Common OmegaConf patterns in the codebase**:
- `OmegaConf.to_container(cfg.nested)` to unpack nested configs as dicts for `**kwargs`
- `OmegaConf.structured(cfg)` to copy configs
- `OmegaConf.create(cfg)` in tests to create configs programmatically

## Key Architecture Patterns

- **CLI modules** in `mussel/cli/` each expose a `main()` function registered as a console script in `pyproject.toml`
- **Model loading** goes through `mussel/models/model_factory.py` which handles all supported foundation models
- **Feature extraction** core logic lives in `mussel/utils/feature_extract.py`
- **Tissue segmentation** is in `mussel/utils/segment.py`
- **File I/O** with remote/cloud support is in `mussel/utils/file.py` (fsspec-based)
- **Model caching** and downloading is handled by `mussel/utils/model_cache.py`

## Dependencies Notes

- PyTorch and TensorFlow are mutually exclusive install extras (see `[tool.uv]` conflicts in `pyproject.toml`)
- Custom packages `transpath` and `timm_ctranspath` come from MSK Mind GitHub repos
- The `fastattn` extra pins specific torch/xformers/flash-attn versions for GigaPath support
- `transformers<4.46` is required for model compatibility
- The `neural-seg` extra installs HEST from GitHub (`mahmoodlab/HEST`) for `seg_model="hest"` support
