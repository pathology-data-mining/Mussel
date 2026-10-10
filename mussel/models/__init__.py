# Import model modules to trigger @register_model self-registration
from . import (abmil, chief, clip, conch, conch_v1, feather, genbio, gigapath,
               googlepath, gpfm, hibou, kaiko, lunit, madeleine, midnight,
               openmidnight, optimus, phikon, prism, resnet, transpath, uni,
               virchow)
from .model_factory import (MODEL_PATCH_SIZES, MODEL_TARGET_MPP,
                            SLIDE_ENCODER_COMPATIBILITY, ModelFactory,
                            ModelType, check_tile_size, get_default_patch_size,
                            get_model_factory, get_required_patch_encoder,
                            recommended_patch_sizes,
                            validate_slide_encoder_compatibility)
