"""Datasets for ImageNet-1k"""
import os, sys
from math import floor
import random

import numpy as np
import torchvision.datasets as dset

if __package__ == None or __package__ == '':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.datasets_base import Base, DATA_ROOT
from dataset.transforms import get_cifar_transforms, get_imgNet_transforms, get_transforms


class ImageNet(Base):
    def __init__(self, split="train", trf=None, trf_type="test", trf_idSet="imagenet", sample_num=None, model="resnet"):
        super().__init__(split, trf_type, trf_idSet)
        self.name = "ImageNet"

        # transform
        if trf is None:
            trf = get_transforms(dataset="imagenet", mode=trf_type, model=model)

        # dataset
        root_path = os.path.join(DATA_ROOT, "imagenet/")
        if split == "train":
            root_path = os.path.join(root_path, "train")
        elif split == "test":
            root_path = os.path.join(root_path, "val")
        self.dataset = dset.ImageFolder(
            root=root_path,
            transform=trf
        )
        self.cls_num = len(self.dataset.classes)

        # id_train number
        self.sample_num = sample_num 
        if self.sample_num is not None:
            self._balance_sample(num_per_class=floor(self.sample_num / self.cls_num), random=True)
    
    def _balance_sample(self, num_per_class, random=True):
        """Sample a balance number of data per class
        It delves into the pytorch ImageFolder class and modify the following members:
            - samples: a list of tuple of (img_file, num_label)
            - imgs: same as samples
            - targets: a list of num_label corr to samples.
        """
        counter_cls = np.array([0] * self.cls_num)
        num_data = len(self.dataset)
        go_through_order = np.arange(num_data)

        if random:
            np.random.shuffle(go_through_order) 

        # go through data
        img_files_new = []
        labels_new = []
        for idx in go_through_order:
            img_file, num_label = self.dataset.samples[idx]  
            if counter_cls[num_label] < num_per_class:
                img_files_new.append(img_file)
                labels_new.append(num_label)
                counter_cls[num_label] += 1

            # stop if the criteria is met
            if np.all(counter_cls >= num_per_class):
                break

        # order the data
        samples_new = [(img_file, num_label) for num_label, img_file in sorted(zip(labels_new, img_files_new))]
        imgs_new = samples_new
        targets_new = [data[1] for data in samples_new] 

        # update the dataset
        self.dataset.samples = samples_new
        self.dataset.imgs = imgs_new
        self.dataset.targets = targets_new
    

if __name__ == "__main__":
    from collections import Counter
    imgnet = ImageNet(id_train_num=200000)
    count = dict(Counter(imgnet.dataset.targets))
    print(f"Min and max population: {min(count.values())} and {max(count.values())}")
