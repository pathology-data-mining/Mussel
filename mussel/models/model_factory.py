import json
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, List

import open_clip
import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torchvision import transforms
from transformers import AutoModel

CFG_DIR = Path(__file__).parent / "configs"

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]


class ModelType(Enum):
    def __init__(self, id, code, path):
        self.id = id
        self.code = code
        self.path = path

    RESNET50 = 1, "resnet50", ""
    CTRANSPATH = 2, "ctranspath", ""
    GIGAPATH = 3, "gigapath", "hf-hub:prov-gigapath/prov-gigapath"
    VIRCHOW = 4, "virchow", "hf-hub:paige-ai/Virchow"
    OPTIMUS = 5, "optimus", "hf-hub:bioptimus/H-optimus-0"
    CLIP = 6, "clip", "hf-hub:wisdomik/QuiltNet-B-16-PMB"
    GOOGLEPATH = 7, "googlepath", "google/path-foundation"
    CONCH1_5 = 8, "conch1_5", "MahmoodLab/TITAN"
    VIRCHOW2 = 9, "virchow2", "hf-hub:paige-ai/Virchow2"


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

    def save(self, save_path: str):
        pass


class GooglePathModel(Model):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        import tensorflow as tf
        from huggingface_hub import from_pretrained_keras

        if model_path is None:
            model_path = ModelType.GOOGLEPATH.path

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

    def save(self, save_path: str):
        raise NotImplementedError("GooglePath model saving not implemented yet")


class TorchModel(Model):
    def __init__(
        self,
        model_path,
        model_obj=None,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if model_obj is None:
            if model_path.startswith("hf-hub:"):
                repo_id = model_path.replace("hf-hub:", "")
                config_file = hf_hub_download(
                    repo_id=repo_id,
                    filename="config.json",
                )
                with open(config_file, "r") as f:
                    config = json.load(f)
                model_name = config.get("model_name", None)
                if not model_name:
                    raise ValueError(
                        f"model_name not found in config.json from {repo_id}"
                    )
                model_obj = timm.create_model(model_name, pretrained=True)
            elif Path(model_path).is_file():
                with open(model_path, "rb") as f:
                    model_obj = pickle.load(f)
            else:
                raise ValueError(f"invalid model_path: {model_path}")
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


class Conch15TorchModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if model_path is None:
            model_path = ModelType.CONCH1_5.path
        model_obj = None
        if not Path(model_path).is_file():
            titan = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            model_obj, _ = titan.return_conch()
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        preprocessing = transforms.Compose(
            [
                transforms.Resize(
                    448,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return preprocessing


class GigapathTorchModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if model_path is None:
            model_path = ModelType.GIGAPATH.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(model_path, pretrained=True)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        preprocessing = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return preprocessing


class OptimusTorchModel(TorchModel):

    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if model_path is None:
            model_path = ModelType.OPTIMUS.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False,
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

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
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if model_path is None:
            model_path = ModelType.VIRCHOW.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj = timm.create_model(
                model_path,
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        preprocessing = create_transform(
            **resolve_data_config(self.obj.pretrained_cfg, model=self.obj)
        )
        return preprocessing


class ClipTorchModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if model_path is None:
            model_path = ModelType.CLIP.path
        model_obj = None
        if model_path.startswith("hf-hub:"):
            model_obj, _, self.preprocessing = open_clip.create_model_and_transforms(
                model_path,
            )
            model_obj.forward = partial(model_obj.encode_image)
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)

    def get_preprocessing_fun(self) -> Callable:
        return self.preprocessing


class TransPathTorchModel(TorchModel):
    def __init__(
        self,
        model_path,
        use_gpu: bool = True,
        gpu_device_id: int | List[int] | None = None,
    ):
        if model_path is None:
            raise ValueError("model_path must be provided for TransPath model")
        from transpath.ctran import ctranspath

        model_obj = ctranspath()
        model_obj.head = nn.Identity()
        td = torch.load(model_path, weights_only=True)
        model_obj.load_state_dict(td["model"], strict=True)
        # ctranspath() module has required torch transforms built in so
        # preprocessing should be None here
        super().__init__(model_path, model_obj, use_gpu, gpu_device_id)


class ResnetTorchModel(TorchModel):
    def __init__(
        self, use_gpu: bool = True, gpu_device_id: int | List[int] | None = None
    ):
        from mussel.models.resnet_custom import resnet50_baseline

        model_obj = resnet50_baseline(pretrained=True)
        super().__init__(None, model_obj, use_gpu, gpu_device_id)


MODEL_FACTORIES = {}


def register_model_factory(model_type: ModelType):
    def decorator(fn):
        MODEL_FACTORIES[model_type] = fn
        return fn

    return decorator


class ModelFactory(ABC):

    @abstractmethod
    def get_model(self, model_path, use_gpu, gpu_device_id) -> Model:
        pass


@register_model_factory(ModelType.GOOGLEPATH)
class GooglePathModelFactory(ModelFactory):
    def get_model(self, model_path, use_gpu=True, gpu_device_id=None) -> Model:
        return GooglePathModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.RESNET50)
class Resnet50ModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        return ResnetTorchModel(use_gpu, gpu_device_id)


@register_model_factory(ModelType.CTRANSPATH)
class CTransPathModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        return TransPathTorchModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.GIGAPATH)
class GigapathModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        return GigapathTorchModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.VIRCHOW)
class VirchowModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        return VirchowTorchModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.VIRCHOW2)
class Virchow2ModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        return VirchowTorchModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.CONCH1_5)
class Conch15ModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        return Conch15TorchModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.OPTIMUS)
class OptimusModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        return OptimusTorchModel(model_path, use_gpu, gpu_device_id)


@register_model_factory(ModelType.CLIP)
class ClipModelFactory(ModelFactory):
    def get_model(self, model_path=None, use_gpu=True, gpu_device_id=None):
        return ClipTorchModel(model_path, use_gpu, gpu_device_id)


def get_model_factory(model_type: ModelType | str = ModelType.CTRANSPATH) -> ModelFactory:
    if isinstance(model_type, str):
        try:
            model_type = ModelType[model_type.upper()]
        except KeyError:
            raise ValueError(f"unknown model type: {model_type}")
    return MODEL_FACTORIES.get(model_type)()
