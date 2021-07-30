""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import OrderedDict

import torch.utils.data as data

import os
import re
import torch
import tarfile
from PIL import Image
import math
import random


IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True, balance=False):
    classes = []
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        if label:
            has_image = False
            for f in files:
                base, ext = os.path.splitext(f)
                if ext.lower() in types:
                    filenames.append(os.path.join(root, f))
                    labels.append(label)
                    has_image = True
            classes.append(label)
            if has_image:
                pass
            else:
                print(f'WARNING: {label} has no image!')
    if class_to_idx is None:
        # building class index
        # unique_labels = set(labels)
        sorted_labels = list(sorted(classes, key=natural_key))
        class_to_idx = OrderedDict()
        for idx, c in enumerate(sorted_labels):
            class_to_idx[c] = idx
    else:
        for cls in classes:
            if cls not in class_to_idx:
                print(f'WARNING: {cls} not in class map!')
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if balance:
        images_and_targets = balance_samples(images_and_targets, len(class_to_idx))
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


def load_class_map(filename, root=''):
    class_map_path = filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, filename)
        assert os.path.exists(class_map_path), 'Cannot locate specified class map file (%s)' % filename
    class_map_ext = os.path.splitext(filename)[-1].lower()
    if class_map_ext == '.txt':
        with open(class_map_path) as f:
            # class_to_idx = {v.strip(): k for k, v in enumerate(f)}
            class_to_idx = OrderedDict()
            for k, v in enumerate(f):
                class_to_idx[v.strip()] = k
    else:
        assert False, 'Unsupported class map extension'
    return class_to_idx


def balance_samples(images_and_targets, n_classes):
    class_counts = [0] * n_classes
    class_images = {}
    for item in images_and_targets:
        class_counts[item[1]] += 1
        class_images.setdefault(item[1], []).append(item[0])
    print('Max class count:', max(class_counts))
    print('Min class count:', min(class_counts))
    thresh = min(2000, max(class_counts))
    print('Balance(square root) class count to:', thresh)
    for class_idx, class_count in enumerate(class_counts):
        if class_count == 0:
            print('WARNING: class {} has no image!'.format(class_idx))
            continue
        # repl_factor = float(thresh) / class_count)  # uniform
        repl_factor = math.sqrt(float(thresh) / class_count)  # square root
        if repl_factor > 1:
            more_count = int((repl_factor - 1) * class_count)
            more_images = random.choices(class_images[class_idx], k=more_count)
            for image in more_images:
                images_and_targets.append((image, class_idx))
    print('Samples after balance:', len(images_and_targets))
    return images_and_targets


class Dataset(data.Dataset):

    def __init__(
            self,
            root,
            load_bytes=False,
            transform=None,
            class_map='',
            balance=False,
            #用onehot vector的方式分类还是聚类的方式
            ):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        #返回的images包含了images和labels
        images, class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx, balance=balance)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        # Others有两种指定方式:
        # 1. 类名中含有“其他”或“Others”关键字
        # 2. 根目录下有others.txt，一个Others类一行
        others_classes = []
        if os.path.exists(os.path.join(root, '../others.txt')):
            with open(os.path.join(root, '../others.txt')) as f:
                for line in f:
                    others_classes.append(line.strip())
        others_indexes = []
        for cls, idx in class_to_idx.items():
            if '其他' in cls or 'Others' in cls or cls in others_classes:
                print('Found other class {} with index {}'.format(cls, idx))
                others_indexes.append(idx)
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform
        self.others_indexes = others_indexes

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        except OSError:
            print(f'Can not open {path}')
            return self.__getitem__(index + 1) if index + 1 < len(self.samples) else self.__getitem__(index - 1)
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    def filenames(self, basename=False, absolute=False):
        fn = lambda x: x
        if basename:
            fn = os.path.basename
        elif not absolute:
            fn = lambda x: os.path.relpath(x, self.root)
        return [fn(x[0]) for x in self.samples]


def _extract_tar_info(tarfile, class_to_idx=None, sort=True):
    files = []
    labels = []
    for ti in tarfile.getmembers():
        if not ti.isfile():
            continue
        dirname, basename = os.path.split(ti.path)
        label = os.path.basename(dirname)
        ext = os.path.splitext(basename)[1]
        if ext.lower() in IMG_EXTENSIONS:
            files.append(ti)
            labels.append(label)
    if class_to_idx is None:
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    tarinfo_and_targets = [(f, class_to_idx[l]) for f, l in zip(files, labels) if l in class_to_idx]
    if sort:
        tarinfo_and_targets = sorted(tarinfo_and_targets, key=lambda k: natural_key(k[0].path))
    return tarinfo_and_targets, class_to_idx


class DatasetTar(data.Dataset):

    def __init__(self, root, load_bytes=False, transform=None, class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        assert os.path.isfile(root)
        self.root = root
        with tarfile.open(root) as tf:  # cannot keep this open across processes, reopen later
            self.samples, self.class_to_idx = _extract_tar_info(tf, class_to_idx)
        self.imgs = self.samples
        self.tarfile = None  # lazy init in __getitem__
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.root)
        tarinfo, target = self.samples[index]
        iob = self.tarfile.extractfile(tarinfo)
        img = iob.read() if self.load_bytes else Image.open(iob).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False):
        filename = self.samples[index][0].name
        if basename:
            filename = os.path.basename(filename)
        return filename

    def filenames(self, basename=False):
        fn = os.path.basename if basename else lambda x: x
        return [fn(x[0].name) for x in self.samples]


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
