import torch
from PIL import Image
import cv2
import io
import json
import segmentation_models_pytorch as smp
import ttach as tta
from pathlib import Path
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .src import archs

class EyeDiseaseSegmentation:
    """
     Class to segment retina lesion, blood vessel and optic disc/cup 
    """
    def __init__(self):      
        CONFIG_AND_CHECKPOINTS = Path(__file__).resolve().parent / "artifact"
        self.transform = self._transform()
        self.model = {
            'EX': self._get_best_model(CONFIG_AND_CHECKPOINTS / 'EX'),
            'HE': self._get_best_model(CONFIG_AND_CHECKPOINTS / 'HE'),
            'SE': self._get_best_model(CONFIG_AND_CHECKPOINTS / 'SE'),
            'MA': self._get_best_model(CONFIG_AND_CHECKPOINTS / 'MA'),
            'OD': self._get_best_model(CONFIG_AND_CHECKPOINTS / 'OD')
        }
    
    def _transform(self):
        resize_transform = [
            A.LongestMaxSize(1024),
            A.PadIfNeeded(min_height=1024, min_width=1024,
                          border_mode=cv2.BORDER_CONSTANT, value=0)
        ]
        preprocessing_fn, self.mean, self.std = archs.get_preprocessing_fn()
        return A.Compose([
            resize_transform,
            A.Lambda(image= preprocessing_fn),
            ToTensorV2()
        ])

    def _get_model(self, params, model_name):
        params['encoder_weights'] = None
        model = getattr(smp, model_name)(
            **params
        )
        return model

    def _get_best_model(self, path):
        checkpoint = torch.load(path/'checkpoints/best.pth')
        with open(path / 'config.json', 'r') as j:
            config = json.load(j)

        if hasattr(smp, config['model_name']):
            model = self._get_model(
                config['model_params'], config['model_name'])
        elif config['model_name'] == "TransUnet":
            from self_attention_cv.transunet import TransUnet
            model = TransUnet(**config['model_params'])
        else:
            model = archs.get_model(
                model_name=config['model_name'], 
                params = config['model_params'],
                training=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = tta.SegmentationTTAWrapper(
                model, tta.aliases.d4_transform(), merge_mode="mean")

        return model

    @torch.no_grad()
    def get_segment(self, image, threshold, lesion_type) -> Image.Image:
        """Return mask overlay on image for particular type of lesion"""
        image_pil = Image.open(io.BytesIO(image)).convert('RGB')
        image_np = np.asarray(image_pil).astype(np.uint8)
        image = self.transform(image=image_np)['image']
        batch_image = image.unsqueeze(0)
        pred = self.model[lesion_type](batch_image)
        pred = torch.sigmoid(pred)[0]
        pred = pred.squeeze(dim=0).numpy()
        mask = (pred > threshold).astype(np.float32)

        im_f = ((image*self.std + self.mean)*255).astype(np.float32)
        mask_color = (0, 255, 0) #Green
        mask = np.expand_dims(mask, axis=-1)
        mask_col = np.expand_dims(np.array(mask_color)/255.0, axis=(0,1))
        overlay = (im_f + 1.0 * mask * (np.mean(0.8 * im_f + 0.2 * 255, axis=2, keepdims=True) * mask_col - im_f)).astype(np.uint8)
        return Image.fromarray(overlay)

