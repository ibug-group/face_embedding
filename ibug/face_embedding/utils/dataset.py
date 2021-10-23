import numbers
import os
import queue as Queue
import threading

# import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .space_warper import WarpingImageToDifferentSpace
from torchvision.datasets import ImageFolder


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


# class MXFaceDataset(Dataset):
#
#     def __init__(
#             self,
#             root_dir,
#             local_rank,
#             project_to_space=None,
#             augment_projection=True,
#             roi_ratio=(0.8, 0.8,),
#             roi_offset_range=(-0.09, 0.09,),
#             angular_offset_range=(-0.35, 0.35,),
#             keep_aspect_ratio=True):
#         """
#         The MXNet face dataset class
#
#         Args:
#             root_dir: the root directory of the dataset files
#             local_rank: the local rank
#             project_to_space: the space to project the image into.
#                 If None, no projection will be used (keep the original space)
#                 Supported space: "roi_tanh_polar","roi_tanh_circular", "roi_tanh"
#             aument_projection: whether to use augmentation for space projection.
#                 If True, random offsets will be added to RoI regions and also the angular offset
#                 If False, no random offsets will be added
#         """
#
#         super(MXFaceDataset, self).__init__()
#         self.root_dir = root_dir
#         self.local_rank = local_rank
#         path_imgrec = os.path.join(root_dir, 'train.rec')
#         path_imgidx = os.path.join(root_dir, 'train.idx')
#         self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
#         s = self.imgrec.read_idx(0)
#         header, _ = mx.recordio.unpack(s)
#         if header.flag > 0:
#             self.header0 = (int(header.label[0]), int(header.label[1]))
#             self.imgidx = np.array(range(1, int(header.label[0])))
#         else:
#             self.imgidx = np.array(list(self.imgrec.keys))
#
#         self.do_projection = False if project_to_space is None else True
#         if project_to_space is not None:  # project image into corresponding, e.g. "RoI Tanh Polar"
#             _, img = mx.recordio.unpack(self.imgrec.read_idx(1))
#             sample = mx.image.imdecode(img).asnumpy()
#             img_size = sample.shape[0:2]  # get image size in the dataset
#
#             # transformation of projecting image into a certain space
#             self.project_to_target_space = WarpingImageToDifferentSpace(
#                 img_size,
#                 img_size,
#                 target_space= project_to_space,
#                 is_training=augment_projection,
#                 roi_ratio=roi_ratio,
#                 roi_offset_range=roi_offset_range,
#                 angular_offset_range=angular_offset_range,
#                 keep_aspect_ratio=keep_aspect_ratio,
#                 squeeze_output=True)
#
#             print('During training, images will be projected into {} space'.format(project_to_space))
#             print("Projection options - RoI Ratio: {}, RoI offset range: {}, "
#                   "Angular offset range: {}, Keep Aspect Ratio: {}".format(
#                 roi_ratio, roi_offset_range, angular_offset_range, keep_aspect_ratio))
#
#             self.transform = transforms.Compose(
#                 [transforms.ToPILImage(),
#                  transforms.RandomHorizontalFlip(),
#                  transforms.ToTensor(),
#                  self.project_to_target_space,   # project into target space
#                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#                  ])
#
#         else:  # the original transformations
#             self.transform = transforms.Compose(
#                 [transforms.ToPILImage(),
#                  transforms.RandomHorizontalFlip(),
#                  transforms.ToTensor(),
#                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#                  ])
#
#     def __getitem__(self, index):
#         idx = self.imgidx[index]
#         s = self.imgrec.read_idx(idx)
#         header, img = mx.recordio.unpack(s)
#         label = header.label
#         if not isinstance(label, numbers.Number):
#             label = label[0]
#         label = torch.tensor(label, dtype=torch.long)
#         sample = mx.image.imdecode(img).asnumpy()
#         if self.transform is not None:
#             sample = self.transform(sample)
#
#         if self.do_projection:
#             rois = self.project_to_target_space.this_roi.squeeze()
#             return sample, label, rois  # also return RoI for information
#         else:
#             return sample, label
#
#     def __len__(self):
#         return len(self.imgidx)


class ImageFolderFaceDataset(Dataset):

    def __init__(
            self,
            root_dir,
            local_rank,):
        """
        The face dataset class for image folders

        Args:
            root_dir: the root directory of the dataset that can be read by torchvision.datasets.ImageFolder
            local_rank: the local rank
        """

        super(ImageFolderFaceDataset, self).__init__()

        self.root_dir = root_dir
        self.local_rank = local_rank

        self.transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])

        self.image_dataset = ImageFolder(root_dir)


    def __getitem__(self, index):

        sample, label = self.image_dataset[index]

        sample = self.transform(sample)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label


    def __len__(self):
        return len(self.image_dataset)

