from torch.utils.data import Subset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import CIFAR10, ImageFolder

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config

Section('cfg', 'arguments to give the writer').params(
    dataset=Param(And(str, OneOf(['cifar', 'imagenet'])), 'Which dataset to write', default='imagenet'),
    split=Param(And(str, OneOf(['train', 'val'])), 'Train or val set', required=True),
    data_dir=Param(str, 'Where to find the PyTorch dataset', required=True),
    write_path=Param(str, 'Where to write the new dataset', required=True),
    write_mode=Param(str, 'Mode: raw, smart or jpg', required=False, default='smart'),
    max_resolution=Param(int, 'Max image side length', required=True),
    num_workers=Param(int, 'Number of workers to use', default=16),
    chunk_size=Param(int, 'Chunk size for writing', default=100),
    jpeg_quality=Param(float, 'Quality of jpeg images', default=90),
    subset=Param(int, 'How many images to use (-1 for all)', default=-1),
    compress_probability=Param(float, 'compress probability', default=None)
)

# from typing import Tuple

import numpy as np
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler

# class ReplaceLabelSymmetric(Operation):
#     """Replace label of specified images.
#     Parameters
#     ----------
#     indices : Sequence[int]
#         The indices of images to relabel.
#     new_label : int
#         The new label to assign.
#     """

#     def __init__(self, indices: int):
#         super().__init__()
#         self.indices = np.sort(indices)
#         # self.new_label = new_label
#         self.classes = list(range(1024))

#     def generate_code(self) -> Callable:

#         to_change = self.indices
#         # new_label = self.new_label
#         class_list = self.classes
#         my_range = Compiler.get_iterator()

#         def replace_label(labels, temp_array, indices):
#             for i in my_range(labels.shape[0]):
#                 sample_ix = indices[i]
#                 position = np.searchsorted(to_change, sample_ix)
#                 if position < len(to_change) and to_change[position] == sample_ix:
#                     # labels[i] = new_label
#                     labels[i] = np.random.choice(list(set(class_list)) - set([labels[i]]))
#             return labels

#         replace_label.is_parallel = True
#         replace_label.with_indices = True

#         return replace_label

#     def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
#         return (replace(previous_state, jit_mode=True), None)

# class CorruptFixedLabels(Operation):
#     def generate_code(self) -> Callable:
#         parallel_range = Compiler.get_iterator()
#         # dst will be None since we don't ask for an allocation
#         def corrupt_fixed(labs, _, inds):
#             for i in parallel_range(labs.shape[0]):
#                 # Because the random seed is tied to the image index, the
#                 # same images will be corrupted every epoch:
#                 np.random.seed(inds[i])
#                 if np.random.rand() < 0.20:
#                     # They will also be corrupted to a deterministic label:
#                     labs[i] = np.random.randint(low=0, high=1024)
#             return labs

#         corrupt_fixed.is_parallel = True
#         corrupt_fixed.with_indices = True
#         return corrupt_fixed

#     def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
#         # No updates to state or extra memory necessary!
#         return previous_state, None

@section('cfg')
@param('dataset')
@param('split')
@param('data_dir')
@param('write_path')
@param('max_resolution')
@param('num_workers')
@param('chunk_size')
@param('subset')
@param('jpeg_quality')
@param('write_mode')
@param('compress_probability')
def main(dataset, split, data_dir, write_path, max_resolution, num_workers,
         chunk_size, subset, jpeg_quality, write_mode,
         compress_probability):
    if dataset == 'cifar':
        my_dataset = CIFAR10(root=data_dir, train=(split == 'train'), download=True)
    elif dataset == 'imagenet':
        my_dataset = ImageFolder(root=data_dir)
    else:
        raise ValueError('Unrecognized dataset', dataset)

    if subset > 0: my_dataset = Subset(my_dataset, range(subset))
    # num_images = 1200000
    # noise_rate = 0.20
    # noisy_indices = np.random.choice(list(range(num_images)), int(noise_rate*num_images)).tolist()
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(write_mode=write_mode,
                               max_resolution=max_resolution,
                               compress_probability=compress_probability,
                               jpeg_quality=jpeg_quality),
        'label': IntField(),
    }, num_workers=num_workers)

    writer.from_indexed_dataset(my_dataset, chunksize=chunk_size)

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
