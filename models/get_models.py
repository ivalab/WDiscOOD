import argparse
import numpy as np
import torch

from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.supcon import SupConResNet
from models.vit import ViT
from models.clip_models import CLIP_RN50


def get_model(arch, args:argparse.Namespace, load_file=None, strict=True):
    """
    Get model

    Arch:
        arch (str):             Architecture name
        args (namespace):       Might contain network settings (e.g. layer number)
        load_file(str):         The parameter file to load. If None, will randomize the parameter
        stric(str):             Strict network parameter loading. N/A if load_file is None
    Return:
        net:                    The neural network model
    """
    arch = arch.lower()

    # create model
    load_pretrain = load_file is None
    if arch == "resnet18":
        net = resnet18(pretrained=load_pretrain, num_classes=args.num_classes)
    elif arch == "resnet34":
        net = resnet34(pretrained=load_pretrain, num_classes=args.num_classes)
    elif arch == "resnet50":
        net = resnet50(pretrained=load_pretrain, num_classes=args.num_classes)
    elif arch == "resnet101":
        net = resnet101(pretrained=load_pretrain, num_classes=args.num_classes)
    elif arch == "resnet50_supcon":
        net = SupConResNet(name="resnet50")
    elif arch == "vit_b":
        assert args.id_dset == "imagenet", "Only implemented the ViT for imagnenet"
        net = ViT('B_16_imagenet1k', pretrained=True)
    elif arch == "resnet50_clip":
        net = CLIP_RN50(device=args.device)
    else:
        raise NotImplementedError(f"The architecture {arch} is not implemented.")

    # load parameter
    # import pdb; pdb.set_trace()
    if load_file is not None:
        # trim state_dict keys for models trained on multi-gpus
        if "supcon" in arch:
            state_dict = torch.load(load_file)["model"]
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    net.encoder = torch.nn.DataParallel(net.encoder)
                else:
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        k = k.replace("module.", "")
                        new_state_dict[k] = v
                    state_dict = new_state_dict
            net.load_state_dict(state_dict, strict=strict)
        else:
            state_dict = torch.load(load_file)
            net.load_state_dict(state_dict, strict=strict)
        
        # load
        print("Model loaded from: {}".format(load_file))
    
    return net
        
   