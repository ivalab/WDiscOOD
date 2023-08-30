import os, sys
from functools import partial
from torch.utils.data import DataLoader


if __package__ == None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.get_datasets import get_dataset



def get_loaders_for_ood(args, trf=None):
    """Get the ID train/test loader and all the OOD loaders 
    This is for the OOD detection evaluation, in the sense that:
    1. The id_train and id_test set won't be shuffled, and the test transform will be applied
    2. The OOD dataloaders will use the id test transform, and will be shuffled for random selection.

    Args:
        trf (torchvision.Transforms):       The transform to apply on all datasets. 
                                            If None, will use default test transforms for the ID dataset based on the architecture used.

    Returns:
        id_train_loader:            The indistribution train loader
        id_test_loader:             The indistribution test loader
        ood_test_loaders (list):    The list of OOD test loaders.
    """
    id_train_dset = get_dataset(args.id_dset, split="train", trf=trf, trf_type="test", trf_idSet=args.id_dset, data_sample_num=args.id_train_num, model=args.arch)
    id_test_dset = get_dataset(args.id_dset, split="test", trf=trf, trf_type="test", trf_idSet=args.id_dset, model=args.arch)
    id_train_loader = DataLoader(id_train_dset, batch_size=args.test_bs, shuffle=True, 
                                num_workers=args.prefetch, pin_memory=True)
    id_test_loader = DataLoader(id_test_dset, batch_size=args.test_bs, shuffle=True,
                                num_workers=args.prefetch, pin_memory=True)

    ood_test_loaders = []
    for ood_name in args.ood_dsets:
        ood_dset = get_dataset(name=ood_name, split="test", trf=trf, trf_type="test", trf_idSet=args.id_dset, model=args.arch)
        ood_loader = DataLoader(ood_dset, batch_size=args.test_bs, shuffle=True,
                                num_workers=args.prefetch, pin_memory=True)
        ood_test_loaders.append(ood_loader)
    
    return id_train_loader, id_test_loader, ood_test_loaders
