"""Datasets for ImageNet-1k"""
import os, sys
import torchvision.datasets as dset

if __package__ == None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.datasets_base import Base, DATA_ROOT, DATASET_ROOT, ImageFilelist
from dataset.transforms import get_cifar_transforms, get_imgNet_transforms, get_transforms

class Textures(Base):
    def __init__(self, split="test", trf=None, trf_type="test", trf_idSet="imagenet", model="resnet"):
        super().__init__()
        self.name = "Textures"

        self.data_dir = os.path.join(DATA_ROOT, "Textures/dtd/images")
        assert os.path.exists(self.data_dir), "Please download the Textures dataset."

        if split == "test":
            if trf is None:
                self.transform = get_transforms(trf_idSet, trf_type, resize_and_crop=True, model=model)
            else:
                self.transform = trf
        else:
            raise NotImplementedError

        self.dataset = dset.ImageFolder(
            root = self.data_dir,
            transform=self.transform
        )

class iNaturalist(Base):
    def __init__(self, split="train", trf=None, trf_type="test", trf_idSet="imagenet", model="resnet"):
        super().__init__(split, trf_type, trf_idSet)
        self.name = "iNaturalist"

        # transform
        if trf is None:
            trf = get_transforms(dataset=trf_idSet, mode=trf_type, model=model)

        # dataset
        root_path = os.path.join(DATA_ROOT, "inaturalist")
        self.dataset = dset.ImageFolder(
            root=root_path,
            transform=trf
        )


class SUN(Base):
    def __init__(self, split="train", trf=None, trf_type="test", trf_idSet="imagenet", model="resnet"):
        super().__init__(split, trf_type, trf_idSet)
        self.name = "SUN"

        # transform
        if trf is None:
            trf = get_transforms(dataset=trf_idSet, mode=trf_type, model=model)

        # dataset
        root_path = os.path.join(DATA_ROOT, "SUN")
        self.dataset = dset.ImageFolder(
            root=root_path,
            transform=trf
        )


class Places(Base):
    def __init__(self, split="train", trf=None, trf_type="test", trf_idSet="imagenet", model="resnet"):
        super().__init__(split, trf_type, trf_idSet)
        self.name = "Places"

        # transform
        if trf is None:
            trf = get_transforms(dataset=trf_idSet, mode=trf_type, model=model)

        # dataset
        root_path = os.path.join(DATA_ROOT, "Places")
        self.dataset = dset.ImageFolder(
            root=root_path,
            transform=trf
        )




class ImageNetO(Base):
    def __init__(self, split="train", trf=None, trf_type="test", trf_idSet="imagenet", model="resnet"):
        super().__init__(split, trf_type, trf_idSet)
        self.name = "ImageNet-O"

        # transform
        if trf is None:
            trf = get_transforms(dataset="imagenet", mode=trf_type, model=model)

        # dataset
        root_path = os.path.join(DATA_ROOT, "imagenet_o")
        self.dataset = dset.ImageFolder(
            root=root_path,
            transform=trf
        )


class OpenImageO(Base):
    def __init__(self, split="test", trf=None, trf_type="test", trf_idSet="imagenet", model="resnet"):
        super().__init__(split, trf_type, trf_idSet)
        self.name = "OpenImage-O"

        # transform
        if trf is None:
            trf = get_transforms(dataset="imagenet", mode=trf_type, model=model)

        # dataset - Need to read from the list
        datalist = os.path.join(DATASET_ROOT, "datalists/openimage_o.txt")
        root_path = os.path.join(DATA_ROOT, f"openimage_o/{split}")
        self.dataset = ImageFilelist(root=root_path, flist=datalist, transform=trf)
    

if __name__ == "__main__":
    dataset = OpenImageO(split="test", trf_type="test", trf_idSet="imagenet")
    import pdb; pdb.set_trace()

