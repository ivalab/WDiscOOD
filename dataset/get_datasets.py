import os, sys

if __package__ == None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.imagenet import ImageNet
from dataset.imagenet_ood import iNaturalist, ImageNetO, OpenImageO, Places, SUN, Textures

DATASETS = {
    "textures": Textures,
    "imagenet": ImageNet,
    "inat": iNaturalist,
    "places": Places,
    "sun": SUN,
    "imagenet_o": ImageNetO,
    "openimage_o": OpenImageO
}

def get_dataset(name:str, split:str, \
    trf=None, trf_type:str="test", trf_idSet:str="imagenet", \
    data_sample_num=None, model="resnet"):
    """
    Get the dataset

    Args:
        name(str):      The dataset name
        split(str):     The split. "train" or "test"
        trf (torchvision.Transforms):       The transform to apply on the dataset. Will overwrite "trf_type" and "try_idSet"
        trf_type(str):  The transform mode. "train" or "test". For inferring the transform if "trf" is absent
        trf_idSet(str):    The in distribution dataset name. For inferring the transform if "trf" is absent
        model (str):    Model could impact the data transform used. Currently it is true for ViT v.s resnet on imagenet
    """ 
    name = name.lower()

    # parse the cifar name
    if name == "cifar" and split == "test":
        if trf_idSet == "cifar10":
            name = "cifar100"
        elif trf_idSet == 'cifar100':
            name = "cifar10"
        else:
            raise NotImplementedError

    assert name in DATASETS.keys(), \
        "The required dataset {} is not supported. Allowed datasets: \n {}".format(name, DATASETS.keys())
    # import pdb;pdb.set_trace()
    if data_sample_num is None:
        return DATASETS[name](split=split, trf=trf, trf_type=trf_type, trf_idSet = trf_idSet, model=model)
    else:
        return DATASETS[name](split=split, trf=trf, trf_type=trf_type, trf_idSet = trf_idSet, model=model, sample_num=data_sample_num)


if __name__ == "__main__":
    import torchvision.transforms as transforms
    import torch
    dset = get_dataset("textures", split="test", trf_type="test", trf_idSet="imagenet")
    trf = torch.nn.Sequential(
        transforms.CenterCrop(10),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    dset_trf = get_dataset("textures", split="test",trf = trf, trf_type="test", trf_idSet="imagenet")
    import pdb;pdb.set_trace()