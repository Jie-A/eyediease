import albumentations as A
import cv2
import numpy as np
from typing import Tuple

__all__ = ['BaseTransform', 
        'EasyTransform', 
        'MediumTransform' ,
        'NormalTransform', 
        'AdvancedTransform', 
        'load_ben_color']
    
class BaseTransform(object):

    def __init__(self, image_size: int = 1024, preprocessing_fn=None):
        self.image_size = image_size
        self.preprocessing_fn = preprocessing_fn

    def pre_transform(self):
        raise NotImplementedError()

    def hard_transform(self):
        raise NotImplementedError()

    def resize_transforms(self):
        raise NotImplementedError()

    def _get_compose(self, transform):
        result = A.Compose([
                item for sublist in transform for item in sublist
            ], 
            additional_targets={
                'mask1': 'mask',
                'mask2': 'mask',
                'mask3': 'mask',
            }
        )
        return result

    def train_transform(self):
        return self._get_compose([
            self.resize_transforms(),
            self.hard_transform()
        ])

    def validation_transform(self):
        return self._get_compose([
            self.pre_transform()
        ])

    def test_transform(self):
        return self.validation_transform()

    def get_preprocessing(self):
        return A.Compose([
            A.Lambda(image=self.preprocessing_fn)
        ])

class NormalTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super(NormalTransform, self).__init__(*args, **kwargs)

    def hard_transform(self):
        return [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.7),
            A.IAAAdditiveGaussianNoise()
        ]
    
    def resize_transforms(self):
        return [
            A.LongestMaxSize(self.image_size),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0)
        ]
    
    def pre_transform(self):
        return self.resize_transforms()

class EasyTransform(NormalTransform):
    def __init__(self, *args, **kwargs):
        super(EasyTransform, self).__init__(*args, **kwargs)

    def hard_transform(self):
        return [
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05,
                                   alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)
            ], p=0.5),
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5)
        ]

class MediumTransform(NormalTransform):
    def __init__(self, *args, **kwargs):
        super(MediumTransform, self).__init__(*args, **kwargs)

    def hard_transform(self):
        return [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.7),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05,
                                   alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)
            ], p=0.5),
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5)
        ]

class AdvancedTransform(NormalTransform):
    def __init__(self, *args, **kwargs):
        super(AdvancedTransform, self).__init__(*args, **kwargs)
    
    def hard_transform(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.7),
            A.OneOf([
                A.RandomContrast(),
                A.RandomGamma(),
                A.RandomBrightness(),
            ]),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ], p=0.0),
            A.ShiftScaleRotate(),
            A.IAAAdditiveGaussianNoise()
        ])

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def load_ben_color(image, mask, sigmaX=10, img_size: Tuple[int] = (256, 256) ):
    mask = crop_image_from_gray(mask)
    mask = cv2.resize(mask, (img_size[0], img_size[1]))
    
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (img_size[0], img_size[1]))
    image=cv2.addWeighted (image, 4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image, mask