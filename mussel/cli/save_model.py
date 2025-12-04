from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import os

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
            # Special handling for CONCH1_5: it's extracted from TITAN_SLIDE, so we save TITAN instead
            if model_type == ModelType.CONCH1_5:
                print(f"\n⊙ {model_type.name} is extracted from TITAN_SLIDE")
                
                # Check if TITAN_SLIDE is already being saved in this batch
                if ModelType.TITAN_SLIDE in cfg.model_types:
                    print(f"  → TITAN_SLIDE will be saved, CONCH1_5 can be extracted from it")
                    continue
                else:
                    # Save TITAN_SLIDE instead
                    print(f"  → Saving TITAN_SLIDE instead (CONCH1_5 can be extracted from it)")
                    model_type = ModelType.TITAN_SLIDE
            
            # Check if model is already downloaded
            model_output_dir = output_dir / model_type.name
            model_output_file = str(model_output_dir) + ".pth"
            
            # Check for directory with .ready marker or .pth file
            is_dir_cached = model_output_dir.is_dir() and (model_output_dir / ".ready").exists()
            is_file_cached = Path(model_output_file).exists()
            
            if is_dir_cached or is_file_cached:
                cache_type = "directory" if is_dir_cached else "file"
                cache_path = model_output_dir if is_dir_cached else model_output_file
                print(f"⊙ {model_type.name} already cached at {cache_path} ({cache_type}), skipping")
                continue
            
            print(f"\nDownloading {model_type.name}...")
            try:
                # Ensure cache directories exist before loading models
                # This prevents "File exists" errors from libraries that don't use exist_ok=True
                cache_dirs = [
                    os.path.expanduser("~/.cache"),
                    os.path.expanduser("~/.cache/huggingface"),
                    os.path.expanduser("~/.cache/torch"),
                ]
                for cache_dir in cache_dirs:
                    os.makedirs(cache_dir, exist_ok=True)
                
                model_factory = get_model_factory(model_type)
                model = model_factory.get_model(None, use_gpu=False)
                
                # Check if model saves as directory (HuggingFace models) or file (pickled models)
                # Try to call save_pretrained if it exists (HuggingFace models)
                if hasattr(model.obj, 'save_pretrained') or hasattr(model, 'save_pretrained'):
                    # HuggingFace model - saves to directory
                    model_output_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        model.save(str(model_output_dir))
                        # Create .ready marker
                        (model_output_dir / ".ready").write_text(f"cached_at={Path(model_output_dir).stat().st_mtime}\n")
                        print(f"✓ {model_type.name} saved to {model_output_dir}/ (directory)")
                    except (NotImplementedError, ValueError) as e:
                        print(f"⊙ {model_type.name} cannot be saved: {e}")
                        print(f"  → Skipping (can be loaded from HuggingFace or extracted from another model)")
                        continue
                else:
                    # Pickled model - saves to file
                    model_output_dir.parent.mkdir(parents=True, exist_ok=True)
                    # Add .pth extension for pickled models
                    try:
                        model.save(model_output_file)
                        print(f"✓ {model_type.name} saved to {model_output_file} (file)")
                    except (NotImplementedError, ValueError) as e:
                        print(f"⊙ {model_type.name} cannot be saved: {e}")
                        print(f"  → Skipping (can be loaded from HuggingFace or extracted from another model)")
                        continue
                    
            except Exception as e:
                print(f"✗ Failed to save {model_type.name}: {e}")
                raise
        
        print(f"\n✓ Successfully saved {len(cfg.model_types)} models to {output_dir}")
        return
    
    # Single model mode
    if not cfg.model_type:
        raise ValueError("Either model_type or model_types must be specified")
    
    if not cfg.output_path:
        raise ValueError("output_path is required when using model_type")
    
    print(f"Downloading {cfg.model_type.name}...")
    model_factory = get_model_factory(cfg.model_type)
    model = model_factory.get_model(cfg.model_path, use_gpu=False)
    model.save(cfg.output_path)
    print(f"✓ Model saved to {cfg.output_path}")


@hydra.main(version_base=None, config_path=".", config_name="save_model_config")
def main(cfg: SaveModelConfig):
    """Download and save a foundation model locally."""
    save_model(cfg)


if __name__ == "__main__":
    main()
