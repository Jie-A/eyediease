import albumentations as A
from albumentations.pytorch import ToTensor


class Transform(object):

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

    def post_transform(self):
        return [A.Lambda(image=self.preprocessing_fn), ToTensor()]

    def _get_compose(self, transform):
        result = A.Compose([
            item for sublist in transform for item in sublist
        ])

        return result

    def train_transform(self):
        return self._get_compose([
            self.resize_transforms(),
            self.hard_transform(),
            self.post_transform()
        ])

    def validation_transform(self):
        return self._get_compose([
            self.pre_transform(),
            self.post_transform()
        ])

    def test_transform(self):
        return self.validation_transform()
