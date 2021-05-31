import torch.nn as nn
import numpy as np
from . import attentionunet, hrnet, doubleunet, dbunet, unet, rcnn_unet, sa_unet

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
    "mobilenetv3_doubleunet": doubleunet.mobilenetv3_doubleunet,
    "vgg_doubleunet": dbunet.DUNet,
    "unet_resnext50_ssl": unet.UneXt50,
    "rrcnn_unet": rcnn_unet.R2U_Net,
    "sa_unet": sa_unet.SA_Unet
}

def get_preprocessing_fn(dataset_name: str):
    if dataset_name == "IDRiD":
        mean = [0.44976714,0.2186806,0.06459363]
        std = [0.33224553,0.17116262,0.086509705]
    elif dataset_name == 'FGADR':
        mean = [0.4554011,0.2591345,0.13285689]
        std = [0.28593522,0.185085,0.13528904]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def preprocessing(x, mean=mean, std=std, **kwargs):
        x = x / 255.0
        if mean is not None:
            mean = np.array(mean)
            x = x - mean

        if std is not None:
            std = np.array(std)
            x = x / std
        return x

    return preprocessing, mean, std

def list_models():
    return list(MODEL_REGISTRY.keys())

def get_model(model_name: str, params=None, training=True) -> nn.Module:   
    try:
        model_fn = MODEL_REGISTRY[model_name.lower()]
    except KeyError:
        raise KeyError(f"Cannot found {model_name}, available options are {list(MODEL_REGISTRY.keys())}")
    if params is None:
        return model_fn()
    if not training:
        params['pretrained'] = False
    return model_fn(**params)
