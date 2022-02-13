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

    def generate_code(self) -> Callable:
        alpha = self.alpha
        same_lam = self.same_lambda
        my_range = Compiler.get_iterator()

        def mixer(images, dst, indices):
            np.random.seed(indices[-1])
            num_images = images.shape[0]
            lam = np.random.beta(alpha, alpha) if same_lam else \
                  np.random.beta(alpha, alpha, num_images)
            for ix in my_range(num_images):
                l = lam if same_lam else lam[ix]
                dst[ix] = l * images[ix] + (1 - l) * images[ix - 1]

            return dst

class Augmix(Operation):

    def generate_code(self):
        my_range = Compiler.get_iterator()
        # dst will be None since we don't ask for an allocation
        def augmix(images, dst, indices):
            num_images = images.shape[0]
            for ix in my_range(num_images):
                dst = (x, aug(x, self.preprocess),
                  aug(x, self.preprocess))

            return dst
        return augmix

    def declare_state_and_memory(self, previous_state) :
        # No updates to state or extra memory necessary!
        return previous_state, None


train_dataset = '/scratch/jcava/imagenet_ffcv/train_400_1.00_50.ffcv'
train_path = Path(train_dataset)

res = self.get_resolution(epoch=0)
self.decoder = RandomResizedCropRGBImageDecoder((res, res))
image_pipeline: List[Operation] = [
    self.decoder,
    Augmix()
]

'''
# order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
order = OrderOption.QUASI_RANDOM
loader = Loader(train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                order=order,
                os_cache=in_memory,
                drop_last=True,
                pipelines={
                    'image': image_pipeline,
                    'label': label_pipeline
                },
                distributed=distributed)
'''
print('Done')