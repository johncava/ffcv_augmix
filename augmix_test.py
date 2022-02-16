import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from torchvision import models
# import torchmetrics
import numpy as np
from tqdm import tqdm

import os
import time
import json
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

from augmix_utils.augmentations import *

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256


from ffcv.pipeline.compiler import Compiler

class Augmix(Operation):

    def __init__(self, jsd=True):
        super().__init__()
        self.jsd =jsd

    def generate_code(self):
        # Compiler.set_enabled(False)
        aug_ = Compiler.compile(aug)
        my_range = Compiler.get_iterator()
        
        def augmix(images, dst):
            num_images = images.shape[0]
            if self.jsd:
                scratch_jsd = ch.zeros(images.shape[0],9,224,224)
                for ix in my_range(num_images):
                    x1 = ch.tensor(np.transpose(images[ix], (2,1,0)))
                    x2 = ch.tensor(np.transpose(aug_(images[ix].copy()),(2,1,0)))
                    x3 = ch.tensor(np.transpose(aug_(images[ix].copy()),(2,1,0)))
                    scratch_jsd[ix] = ch.cat((x1,x2,x3),0)
                return scratch_jsd
            else:
                scratch = ch.zeros(images.shape[0],3,224,224)
                for ix in my_range(num_images):
                    scratch[ix] = ch.tensor(np.transpose(aug_(images[ix]), (2,1,0)))
                return scratch
        return augmix

    def declare_state_and_memory(self, previous_state) :
        
        return previous_state, None


train_dataset = '/scratch/jcava/imagenet_ffcv/train_400_1.00_50.ffcv'
train_path = Path(train_dataset)

decoder = RandomResizedCropRGBImageDecoder((224, 224))
image_pipeline: List[Operation] = [
    decoder,
    Augmix(jsd=False)
]


label_pipeline: List[Operation] = [
    IntDecoder(),
    ToTensor()
]

distributed = 0

# order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
order = OrderOption.QUASI_RANDOM
loader = Loader(train_dataset,
                batch_size=1024,
                num_workers=12,
                order=order,
                os_cache=1,
                drop_last=True,
                pipelines={
                    'image': image_pipeline,
                    'label': label_pipeline
                },
                distributed=distributed)

for ims, labs in loader:
    # ims = ch.split(ims, 3, dim=1)
    # print(ims[0].size())
    print(ims.size())
    break
print('Done')
