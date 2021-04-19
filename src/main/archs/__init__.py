import torch.nn as nn
import numpy as np
from . import attentionunet, hrnet, doubleunet

__all__ = ['list_models', 'get_model', 'get_preprocessing_fn']

MODEL_REGISTRY = {
    "resnet50_attunet": attentionunet.resnet50_attunet, 
    "efficientnetb2_attunet": attentionunet.efficientnetb2_attunet,
    "mobilenetv3_attunet": attentionunet.mobilenetv3_attunet,
    "hrnet18": hrnet.hrnet18, 
    "hrnet34": hrnet.hrnet34, 
    "hrnet48": hrnet.hrnet48,
    "resnet50_doubleunet": doubleunet.resnet50_doubleunet,
    "efficientnetb2_doubleunet": doubleunet.efficientnetb2_doubleunet,
    "mobilenetv3_doubleunet": doubleunet.mobilenetv3_doubleunet   
}

def get_preprocessing_fn(pretrained="imagenet"):
    if pretrained == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean, std = None, None

    def preprocessing(x, mean=mean, std=std, **kwargs):
        x = x / 255.0
        if mean is not None:
            mean = np.array(mean)
            x = x - mean

        if std is not None:
            std = np.array(std)
            x = x / std
        return x

    return preprocessing

def list_models():
    return list(MODEL_REGISTRY.keys())

def get_model(model_name: str, params) -> nn.Module:   
    try:
        model_fn = MODEL_REGISTRY[model_name.lower()]
    except KeyError:
        raise KeyError(f"Cannot found {model_name}, available options are {list(MODEL_REGISTRY.keys())}")

    return model_fn(**params), get_preprocessing_fn()
