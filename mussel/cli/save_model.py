from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from mussel.models.model_factory import ModelType, get_model_factory


@dataclass
class SaveModelConfig:
    """
    model_type (ModelType): Type of model to save.
    model_path (Optional[str]): Path to the model weights.
    output_path (str): Path to save the model.
    """

    model_type: ModelType = MISSING
    model_path: Optional[str] = None
    output_path: str = MISSING


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
    """Load and save a foundation model to disk.
    
    Args:
        cfg: Configuration with model_type, model_path, and output_path.
    """
    model_factory = get_model_factory(cfg.model_type)
    model = model_factory.get_model(cfg.model_path, use_gpu=False)
    model.save(cfg.output_path)
    print(f"Model saved to {cfg.output_path}")


@hydra.main(version_base=None, config_path=".", config_name="save_model_config")
def main(cfg: SaveModelConfig):
    """Download and save a foundation model locally."""
    save_model(cfg)


if __name__ == "__main__":
    main()
