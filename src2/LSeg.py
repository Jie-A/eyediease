import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.vgg import vgg16


class LSeg(nn.Module):
    def __init__(self, in_ch, classes, dropout, depth):
        pass
    