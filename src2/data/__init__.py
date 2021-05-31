from .lesion_dataset import *
from .data_transform import *

def get_transform(name):
    if name == 'normal':
        return NormalTransform
    if name == 'easy':
        return EasyTransform
    if name == 'medium':
        return MediumTransform
    if name == 'advanced':
        return AdvancedTransform