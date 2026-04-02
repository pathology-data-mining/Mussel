import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from mussel.models.model_factory import ModelType, get_model_factory


@dataclass
class SaveModelConfig:
    """
    Single Model Mode:
        model_type (ModelType): Type of model to save.
        model_path (Optional[str]): Path to the model weights.
        output_path (str): Path to save the model.

    Multi-Model Mode:
        model_types (List[ModelType]): List of model types to save.
        model_dir (str): Directory to save all models (each in subdirectory named after model type).
    """

    model_type: Optional[ModelType] = None
    model_path: Optional[str] = None
    output_path: Optional[str] = None

    # Multi-model mode
    model_types: Optional[List[ModelType]] = None
    model_dir: Optional[str] = None


desc_doc = """== ${hydra.help.app_name} ==

Save a machine learning model to a specified path.
"""

parameter_doc = f"""== Available Parameters ==
{SaveModelConfig.__doc__}
"""

cs = ConfigStore.instance()
cs.store(
    group="hydra",
    name="config",
    node=HydraConf(help=HelpConf(header=desc_doc, footer=parameter_doc)),
    provider="hydra",
)
cs.store(name="save_model_config", node=SaveModelConfig)


def _ensure_cache_dirs() -> None:
    """Create standard HuggingFace/torch cache dirs unless overridden by env vars."""
    if os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE"):
        return
    for raw in ("~/.cache", "~/.cache/huggingface", "~/.cache/torch"):
        p = Path(os.path.expanduser(raw))
        (p.resolve() if p.is_symlink() else p).mkdir(parents=True, exist_ok=True)


def _save_one_model(model_type: ModelType, output_dir: Path) -> None:
    """Download and save one model into output_dir, skipping if already cached."""
    model_dir = output_dir / model_type.name
    model_file = Path(str(model_dir) + ".pth")

    if (model_dir / ".ready").exists():
        print(f"⊙ {model_type.name} already cached at {model_dir}, skipping")
        return
    if model_file.exists():
        print(f"⊙ {model_type.name} already cached at {model_file}, skipping")
        return

    print(f"\nDownloading {model_type.name}...")
    _ensure_cache_dirs()
    model = get_model_factory(model_type).get_model(None, use_gpu=False)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.save(str(model_dir))
        (model_dir / ".ready").write_text("cached\n")
        print(f"✓ {model_type.name} saved to {model_dir}/")
    except (NotImplementedError, ValueError):
        try:
            model.save(str(model_file))
            print(f"✓ {model_type.name} saved to {model_file}")
        except (NotImplementedError, ValueError) as e:
            print(f"⊙ {model_type.name} cannot be saved locally: {e}")


def save_model(cfg: SaveModelConfig):
    """Load and save foundation model(s) to disk.

    Args:
        cfg: Configuration for single or multi-model saving.
    """
    # Multi-model mode
    if cfg.model_types is not None:
        if not cfg.model_dir:
            raise ValueError("model_dir is required when using model_types")
        output_dir = Path(cfg.model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(cfg.model_types)} models to {output_dir}")

        for model_type in cfg.model_types:
            # CONCH1_5 is extracted from TITAN_SLIDE; save TITAN_SLIDE instead
            if model_type == ModelType.CONCH1_5:
                print(f"\n⊙ {model_type.name} is extracted from TITAN_SLIDE")
                if ModelType.TITAN_SLIDE in cfg.model_types:
                    print(
                        f"  → TITAN_SLIDE will be saved; CONCH1_5 can be extracted from it"
                    )
                    continue
                print(
                    f"  → Saving TITAN_SLIDE instead (CONCH1_5 can be extracted from it)"
                )
                model_type = ModelType.TITAN_SLIDE

            try:
                _save_one_model(model_type, output_dir)
            except Exception as e:
                print(f"✗ Failed to save {model_type.name}: {e}")
                raise

        print(f"\n✓ Done saving models to {output_dir}")
        return

    # Single model mode
    if not cfg.model_type:
        raise ValueError("Either model_type or model_types must be specified")
    if not cfg.output_path:
        raise ValueError("output_path is required when using model_type")

    print(f"Downloading {cfg.model_type.name}...")
    model = get_model_factory(cfg.model_type).get_model(cfg.model_path, use_gpu=False)
    model.save(cfg.output_path)
    print(f"✓ Model saved to {cfg.output_path}")


@hydra.main(version_base=None, config_path=".", config_name="save_model_config")
def main(cfg: SaveModelConfig):
    """Download and save a foundation model locally."""
    save_model(cfg)


if __name__ == "__main__":
    main()
