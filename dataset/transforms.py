from torchvision import transforms

# [mean, std]
DATASET_STATS = {
    # Training stats
    "cifar10": [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    "cifar100": [[0.507, 0.487, 0.441], [0.267, 0.256, 0.276]],
    "tin": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    # Testing (open world) stats
    "test": [[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
}


def get_cifar_transforms(dataset="cifar100", mode="test", resize_and_crop=True):
    """Get the Cifar data transform

    Args:
        resize_and_crop (bool):         Option to add the resize and crop in the transform. For now only valid for the test mode
                                        This might not be necessary for the OOD tests size of the same resolution as the in-distribution CIFAR (32-by-32)
    """
    assert "cifar" in dataset, "Only for cifar transforms, but the desired dataset is {}".format(dataset)
    mean, std = DATASET_STATS[dataset]
    img_res = 32

    if resize_and_crop:
        transform_cifar_test = transforms.Compose([
            transforms.Resize((img_res, img_res)),
            transforms.CenterCrop(img_res),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # transforms.Normalize([x/255.0 for x in [125.3, 123.0, 113.9]],
            #                     [x/255.0 for x in [63.0, 62.1, 66.7]]),
        ])
    else:
        transform_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # transforms.Normalize([x/255.0 for x in [125.3, 123.0, 113.9]],
            #                     [x/255.0 for x in [63.0, 62.1, 66.7]]),
        ])


    transform_cifar_train = transforms.Compose([
        # transforms.RandomCrop(imagesize, padding=4),
        transforms.RandomResizedCrop(size=img_res, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # transforms.Normalize([x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                      [x / 255.0 for x in [63.0, 62.1, 66.7]]),
    ])


    return eval("transform_cifar_{}".format(mode))


def get_imgNet_transforms(mode="test"):
    # Follows the one here: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L236-L252
    transform_imgNet_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_imgNet_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return eval("transform_imgNet_{}".format(mode))


def get_imgNet_transforms_vit(mode="test"):
    # The chosen B-16_imagenet21 operates on a different resolution. 
    # See: https://github.com/lukemelas/PyTorch-Pretrained-ViT
    transform_imgNet_test_vit =  transforms.Compose([
        transforms.Resize((384, 384)), 
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    # NOTE: not sure if the train transform is correct. But it won't be used.
    transform_imgNet_train_vit = transforms.Compose([
        transforms.Resize(384),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return eval("transform_imgNet_{}_vit".format(mode))


def get_transforms(dataset="cifar100", mode="test", resize_and_crop=False, model="resnet"):
    """Get the data tranform

    Args:
        dataset (str, optional):    The transform targeting which dataset.
                                    Choices: ["cifar10", "cifar100", "imagenet"]. Defaults to "cifar".
        mode (str, optional):       "train" or "test". Defaults to "test".
    """

    if "cifar" in dataset:
        return get_cifar_transforms(dataset, mode=mode, resize_and_crop=resize_and_crop)
    elif dataset == "imagenet":
        if ("vit" in model and "clip" not in model):
            return get_imgNet_transforms_vit(mode)
        else:
            return get_imgNet_transforms(mode)
    else:
        raise NotImplementedError
    

if __name__ == "__main__":

    print(get_transforms("cifar10", mode="test"))
    print(get_transforms("cifar10", mode="train"))

    print(get_transforms("cifar100", mode="test"))
    print(get_transforms("cifar100", mode="train"))


    print(get_transforms("imgNet", mode="test"))
    print(get_transforms("imgNet", mode="train"))