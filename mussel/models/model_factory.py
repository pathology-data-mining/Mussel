from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List
from pathlib import Path
import json

import open_clip
import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torchvision import transforms
from conch.open_clip_custom import load_checkpoint, CoCa, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from conch.open_clip_custom.factory import read_state_dict
from conch.open_clip_custom.transform import image_transform
from huggingface_hub import hf_hub_download

CFG_DIR = Path(__file__).parent / 'configs'

class ModelType(Enum):
    def __init__(self, id, code, hf_path):
        self.id = id
        self.code = code
        self.hf_path = hf_path

    RESNET50 = 1, "resnet50", ""
    CTRANSPATH = 2, "ctranspath", ""
    GIGAPATH = 3, "gigapath", "hf-hub:prov-gigapath/prov-gigapath"
    VIRCHOW = 4, "virchow", "hf-hub:paige-ai/Virchow"
    OPTIMUS = 5, "optimus", "hf-hub:bioptimus/H-optimus-0"
    CLIP = 6, "clip", "hf-hub:wisdomik/QuiltNet-B-16-PMB"
    GOOGLEPATH = 7, "googlepath", "google/path-foundation"
    CONCH = 9, "conch", "hf-hub:MahmoodLab/CONCH"
    CONCH1_5 = 10, "conch1_5", "hf-hub:MahmoodLab/conchv1_5"
    VIRCHOW2 = 11, "virchow2", "hf-hub:paige-ai/Virchow2"


class Model:
    def __init__(
        self,
        model_obj,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        self.obj = model_obj

    def get_model_fun(self) -> Callable:
        return self.obj

    def get_preprocessing_fun(self) -> Callable:
        return None


class GooglePathModel(Model):
    def __init__(
        self,
        model_path,
        model_obj=None,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        import tensorflow as tf

        if use_gpu and len(tf.config.list_physical_devices("GPU")) == 0:
            raise OSError("cuda not available")
        if use_gpu and gpu_device_id:
            if isinstance(gpu_device_id, list):
                devices = [
                    tf.config.list_physical_devices("GPU")[i] for i in gpu_device_id
                ]
            else:
                devices = tf.config.list_physical_devices("GPU")[gpu_device_id]
            tf.config.set_visible_devices(devices)

        if not model_obj:
            from huggingface_hub import from_pretrained_keras

            model_obj = from_pretrained_keras(model_path)

        super().__init__(model_obj)

    def get_model_fun(self) -> Callable:
        import tensorflow as tf

        def model_fun(x) -> Callable:
            tensor = tf.cast(x, tf.float32) / 255.0
            tensor = tf.transpose(tensor, [0, 2, 3, 1])
            tensor = tf.image.resize(
                tensor, size=(224, 224), method=tf.image.ResizeMethod.BICUBIC
            )
            return self.obj(tensor)

        return model_fun


class TorchModel(Model):
    def __init__(
        self,
        model_obj,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        super().__init__(model_obj)
        if use_gpu and not torch.cuda.is_available():
            raise OSError("cuda not available")

        device_type = "cuda" if use_gpu else "cpu"
        device_id = (
            gpu_device_id[0]
            if isinstance(gpu_device_id, list) and len(gpu_device_id) > 0
            else gpu_device_id
        )
        self.device = (
            torch.device(device_type, device_id)
            if device_id is not None
            else torch.device(device_type)
        )

        if isinstance(gpu_device_id, list) and len(gpu_device_id) > 1:
            self.obj = nn.DataParallel(self.obj, device_ids=gpu_device_id)
        self.obj = self.obj.to(self.device)
        self.obj.eval()

    def get_model_fun(self) -> Callable:
        def model_fun(x):
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
            ):
                x = x.to(self.device, non_blocking=True)
                return self.obj(x).cpu()

        return model_fun

class ConchModel(TorchModel):

    def __init__(
        self,
        model_path,
        model_obj=None,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        model_cfg_path = CFG_DIR / "conch_ViT-B-16.json"
        with open(model_cfg_path, 'r') as f:
            model_cfg = json.load(f)

        _ = model_cfg.pop('custom_text', None)
        model_obj = CoCa(**model_cfg)

        if model_path.startswith("hf-hub:"): 
            _ = hf_hub_download(model_path[len("hf-hub:"):], 
                                filename="meta.yaml")
            model_path = hf_hub_download(model_path[len("hf-hub:"):], 
                                              filename="pytorch_model.bin")

        load_checkpoint(model_obj, model_path)

        model_obj.visual.image_mean = OPENAI_DATASET_MEAN
        model_obj.visual.image_std = OPENAI_DATASET_STD

        super().__init__(model_obj, use_gpu, gpu_device_id)


    def get_preprocessing_fun(self) -> Callable:
        image_mean = getattr(self.obj.visual, 'image_mean', None)
        image_std = getattr(self.obj.visual, 'image_std', None)

        return image_transform(
            self.obj.visual.image_size,
            mean=image_mean,
            std=image_std,
        )

    def get_model_fun(self) -> Callable:
        def model_fun(x):
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
            ):
                x = x.to(self.device, non_blocking=True)
                return self.obj.encode_image(x).cpu()

        return model_fun

class Conch15Model(ConchModel):
    def __init__(
        self,
        model_path,
        model_obj=None,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        model_cfg_path = CFG_DIR / "conch_ViT-B-16.json"
        with open(model_cfg_path, 'r') as f:
            model_cfg = json.load(f)

        _ = model_cfg.pop('custom_text', None)
        model_obj = CoCa(**model_cfg)
        if model_path.startswith("hf-hub:"): 
            _ = hf_hub_download(model_path[len("hf-hub:"):], 
                                filename="meta.yaml")
            model_path = hf_hub_download(model_path[len("hf-hub:"):], 
                                         filename="pytorch_model_vision.bin")

        state_dict = read_state_dict(model_path)
        model_obj.load_state_dict(state_dict, strict=False)

        TorchModel.__init__(self, model_obj, use_gpu, gpu_device_id)


class GigapathTorchModel(TorchModel):
    def __init__(
        self,
        model_path,
        model_obj=None,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if model_obj is None:
            model_obj = timm.create_model(model_path, pretrained=True)
        super().__init__(model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        preprocessing = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return preprocessing


class OptimusTorchModel(TorchModel):

    def __init__(
        self,
        model_path,
        model_obj=None,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if not model_obj:
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False,
            )
        super().__init__(model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        preprocessing = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
        return preprocessing


class VirchowTorchModel(TorchModel):
    def __init__(
        self,
        model_path,
        model_obj=None,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if not model_obj:
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        super().__init__(model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        preprocessing = create_transform(
            **resolve_data_config(self.obj.pretrained_cfg, model=self.obj)
        )
        return preprocessing


class ClipTorchModel(TorchModel):
    def __init__(
        self,
        model_path,
        model_obj=None,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        model_obj, _, self.preprocessing = open_clip.create_model_and_transforms(
            model_path,
        )
        super().__init__(model_obj, use_gpu, gpu_device_id)

    def get_model_fun(self) -> Callable:
        def model_fun(x):
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
            ):
                x = x.to(self.device, non_blocking=True)
                return self.obj.encode_image(x).cpu()

        return model_fun

    def get_preprocessing_fun(self) -> Callable:
        return self.preprocessing


class TransPathTorchModel(TorchModel):
    def __init__(
        self,
        model_path,
        model_obj=None,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if not model_obj:
            from transpath.ctran import ctranspath

            model_obj = ctranspath()
            model_obj.head = nn.Identity()
            td = torch.load(model_path, weights_only=True)
            model_obj.load_state_dict(td["model"], strict=True)
        # ctranspath() module has required torch transforms built in so
        # preprocessing should be None here
        super().__init__(model_obj, use_gpu, gpu_device_id)


class ResnetTorchModel(TorchModel):
    def __init__(
        self, use_gpu: bool = True, gpu_device_id: int | List[int] | None = None
    ):
        from mussel.models.resnet_custom import resnet50_baseline

        model_obj = resnet50_baseline(pretrained=True)
        super().__init__(model_obj, use_gpu, gpu_device_id)


MODEL_FACTORIES = {}


def register_model_factory(model_type: ModelType):
    def decorator(fn):
        MODEL_FACTORIES[model_type] = fn
        return fn

    return decorator


class ModelFactory(ABC):

    @abstractmethod
    def get_model(self, model_path, model_obj, use_gpu, gpu_device_id) -> Model:
        pass


@register_model_factory(ModelType.GOOGLEPATH)
class GooglePathModelFactory(ModelFactory):
    def get_model(
        self, model_path, model_obj=None, use_gpu=True, gpu_device_id=None
    ) -> Model:
        return GooglePathModel(model_path, model_obj, use_gpu, gpu_device_id)


@register_model_factory(ModelType.RESNET50)
class Resnet50ModelFactory(ModelFactory):
    def get_model(
        self, model_path=None, model_obj=None, use_gpu=True, gpu_device_id=None
    ):
        return ResnetTorchModel(use_gpu, gpu_device_id)


@register_model_factory(ModelType.CTRANSPATH)
class CTransPathModelFactory(ModelFactory):
    def get_model(
        self, model_path=None, model_obj=None, use_gpu=True, gpu_device_id=None
    ):
        return TransPathTorchModel(model_path, model_obj, use_gpu, gpu_device_id)


@register_model_factory(ModelType.GIGAPATH)
class GigapathModelFactory(ModelFactory):
    def get_model(
        self, model_path=None, model_obj=None, use_gpu=True, gpu_device_id=None
    ):
        return GigapathTorchModel(model_path, model_obj, use_gpu, gpu_device_id)


@register_model_factory(ModelType.VIRCHOW)
class VirchowModelFactory(ModelFactory):
    def get_model(
        self, model_path=None, model_obj=None, use_gpu=True, gpu_device_id=None
    ):
        return VirchowTorchModel(model_path, model_obj, use_gpu, gpu_device_id)

@register_model_factory(ModelType.VIRCHOW2)
class Virchow2ModelFactory(ModelFactory):
    def get_model(
        self, model_path=None, model_obj=None, use_gpu=True, gpu_device_id=None
    ):
        return VirchowTorchModel(model_path, model_obj, use_gpu, gpu_device_id)

@register_model_factory(ModelType.CONCH)
class ConchModelFactory(ModelFactory):
    def get_model(
        self, model_path=None, model_obj=None, use_gpu=True, gpu_device_id=None
    ):
        return ConchModel(model_path, model_obj, use_gpu, gpu_device_id)


@register_model_factory(ModelType.CONCH1_5)
class Conch15ModelFactory(ModelFactory):
    def get_model(
        self, model_path=None, model_obj=None, use_gpu=True, gpu_device_id=None
    ):
        return Conch15Model(model_path, model_obj, use_gpu, gpu_device_id)

@register_model_factory(ModelType.OPTIMUS)
class OptimusModelFactory(ModelFactory):
    def get_model(
        self, model_path=None, model_obj=None, use_gpu=True, gpu_device_id=None
    ):
        return OptimusTorchModel(model_path, model_obj, use_gpu, gpu_device_id)


@register_model_factory(ModelType.CLIP)
class ClipModelFactory(ModelFactory):
    def get_model(
        self, model_path=None, model_obj=None, use_gpu=True, gpu_device_id=None
    ):
        return ClipTorchModel(model_path, model_obj, use_gpu, gpu_device_id)


def get_model_factory(model_type: ModelType = ModelType.CTRANSPATH) -> ModelFactory:
    return MODEL_FACTORIES.get(model_type)()
