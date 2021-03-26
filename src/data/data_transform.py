import albumentations as A
from albumentations.pytorch import ToTensor
import cv2

__all__ = ['BaseTransform', 'EasyTransform', 'MediumTransform' ,'NormalTransform']

class BaseTransform(object):

    def __init__(self, image_size: int = 1024, preprocessing_fn=None):
        self.image_size = image_size
        self.preprocessing_fn = preprocessing_fn

    def pre_transform(self):
        return [
            A.Resize(self.image_size,
                     self.image_size, always_apply=True)
        ]

    def hard_transform(self):
        result = [
            A.RandomRotate90(),
            A.Cutout(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            ),
            A.GridDistortion(p=0.3),
            A.HueSaturationValue(p=0.3)
        ]

        return result

    def resize_transforms(self):
        pre_size = int(self.image_size * 2)

        random_crop = A.Compose([
            A.SmallestMaxSize(pre_size, p=1),
            A.RandomCrop(
                self.image_size, self.image_size, p=1
            )
        ])

        rescale = A.Compose(
            [A.Resize(self.image_size, self.image_size, p=1)])

        random_crop_big = A.Compose([
            A.LongestMaxSize(pre_size, p=1),
            A.RandomCrop(
                self.image_size, self.image_size, p=1
            )
        ])

        # Converts the image to a square of size self.image_size x self.image_size
        result = [
            A.OneOf([
                random_crop,
                rescale,
                random_crop_big
            ], p=1)
        ]

        return result

    def _get_compose(self, transform):
        result = A.Compose([
            item for sublist in transform for item in sublist
        ])

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

class EasyTransform(BaseTransform):
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

    def resize_transforms(self):
        return [
            A.LongestMaxSize(self.image_size),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0)
        ]
    
    def pre_transform(self):
        return self.resize_transforms()

class MediumTransform(EasyTransform):
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


class NormalTransform(EasyTransform):
    def __init__(self, *args, **kwargs):
        super(NormalTransform, self).__init__(*args, **kwargs)

    def hard_transform(self):
        return [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.7),
            # A.OneOf([
            #     A.ElasticTransform(alpha=120, sigma=120 * 0.05,
            #                        alpha_affine=120 * 0.03, p=0.5),
            #     A.GridDistortion(p=0.5),
            #     A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)
            # ], p=0.5),
        ]
