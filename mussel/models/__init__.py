# Import model modules to trigger @register_model self-registration
from . import (abmil, chief, clip, conch, conch_v1, feather, genbio, gigapath,
               googlepath, gpfm, hibou, kaiko, lunit, madeleine, midnight,
               openmidnight, optimus, phikon, prism, resnet, transpath, uni,
               virchow)
from .model_factory import (MODEL_PATCH_SIZES, SLIDE_ENCODER_COMPATIBILITY,
                            ModelFactory, ModelType, get_default_patch_size,
                            get_model_factory, get_required_patch_encoder,
                            validate_slide_encoder_compatibility)
