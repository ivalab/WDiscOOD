"""
Extract and save the features for the OOD detections.
Adopted some code from the knn-ood: https://github.com/deeplearning-wisc/knn-ood

"""
import pdb
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset

from dataset.dataloaders import get_loaders_for_ood
from models.get_models import get_model
from utils.argparser import FeatExtractArgs
from utils.utils import get_feat_dims


argparser = FeatExtractArgs()
args = argparser.get_args()

# ========================== Prepare

# model. Enforce strict here.
net = get_model(arch=args.arch, args=args, load_file=args.load_file, strict=True)
net.eval()
if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
elif args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"

cudnn.benchmark = True  # fire on all cylinders


# the data loaders
if "clip" in args.arch:
    # use clip transform
    transform = net.get_transform()
    id_train_loader, id_test_loader, ood_loaders = get_loaders_for_ood(args, trf=transform)
    # print(f"The clip transform: {transform}")
    # for loader in ood_loaders + [id_train_loader, id_test_loader]:
    #     print(f"The transform for {loader.dataset.name} dataset: {loader.dataset.dataset.transform}")
else:
    id_train_loader, id_test_loader, ood_loaders = get_loaders_for_ood(args)

# embed mode - no classifier is used for the network. Rather, an additional embed space is appended.
embed_mode = "supcon" in args.arch or "clip" in args.arch

# get the feature dimensions
proj_dim, featdims = get_feat_dims(args, net, embed_mode=embed_mode)

# prepare the folder
save_folder = os.path.join(args.save_folder, args.id_dset)
save_folder = os.path.join(save_folder, f"{args.arch}")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# ======================= Save features of all the indistribution data
for split, in_loader in [('train', id_train_loader), ('test', id_test_loader),]:
# for split, in_loader in [('test', id_test_loader)]:

    print(f"Extracting the features of {args.id_dset}-{split}...")

    data_num = len(in_loader.dataset)

    if args.large_scale and args.rerun:
        feat_log_name = f"{save_folder}/id_{split}_feat.mmap"
        label_log_name = f"{save_folder}/id_{split}_label.mmap"
        logit_or_projFeat_log_name = f"{save_folder}/id_{split}_proj.mmap" if embed_mode else f"{save_folder}/id_{split}_logit.mmap" 
        if split == "train":
            feat_log_name = f"{feat_log_name.split('.')[0]}_{args.id_train_num}.mmap"
            label_log_name = f"{label_log_name.split('.')[0]}_{args.id_train_num}.mmap"
            logit_or_projFeat_log_name = f"{logit_or_projFeat_log_name.split('.')[0]}_{args.id_train_num}.mmap" 
        feat_log = np.memmap(feat_log_name, dtype='float32', mode='w+', shape=(data_num, featdims[-1]))
        label_log = np.memmap(label_log_name, dtype='float32', mode='w+', shape=(data_num,))
        logit_or_projFeat_log = np.memmap(logit_or_projFeat_log_name, dtype='float32', mode='w+', shape=(data_num, proj_dim)) if embed_mode \
            else np.memmap(logit_or_projFeat_log_name, dtype='float32', mode='w+', shape=(data_num, args.num_classes))
    else:
        feat_log = np.zeros((data_num, sum(featdims)))
        logit_log = np.zeros((data_num, args.num_classes))
        label_log = np.zeros(data_num)
        save_name = f"{save_folder}/id_{split}_alllayers.npy"

    if args.rerun:
        net.eval()
        with torch.no_grad():
            with tqdm(total=data_num) as pbar:
                for batch_idx, (inputs, targets) in enumerate(in_loader):
                    # import pdb; pdb.set_trace()
                    # if batch_idx >= 1:
                        # break
                    inputs, targets = inputs.to(device), targets.to(device)
                    start_ind = batch_idx * args.test_bs
                    end_ind = min((batch_idx + 1) * args.test_bs, len(in_loader.dataset))

                    # NOTE: by default the supcon model WON'T normalize feature
                    logit_or_projFeat, feature_list = net.feature_list(inputs)

                    # save all features for small scale, but last feature for large scale
                    if args.large_scale:
                        feat = F.adaptive_avg_pool2d(feature_list[-1], 1).squeeze()
                    else:
                        feat = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)

                    feat_log[start_ind:end_ind, :] = feat.data.cpu().numpy()
                    label_log[start_ind:end_ind] = targets.data.cpu().numpy()
                    logit_or_projFeat_log[start_ind:end_ind] = logit_or_projFeat.data.cpu().numpy()

                    pbar.update(args.test_bs)

                    if end_ind >= data_num:
                        break
        
            if not args.large_scale:
                np.save(save_name, (feat_log.T, logit_log.T, label_log))
    else:
        if args.large_scale:
            feat_log = np.memmap(f"{save_folder}/id_{split}_feat_{args.id_train_num}.mmap", dtype='float32', mode='r', shape=(len(in_loader.dataset), featdims[-1]))
            logit_log = np.memmap(f"{save_folder}/id_{split}_logit_{args.id_train_num}.mmap", dtype='float32', mode='r', shape=(len(in_loader.dataset), args.num_classes))
            label_log = np.memmap(f"{save_folder}/id_{split}_label_{args.id_train_num}.mmap", dtype='float32', mode='r', shape=(len(in_loader.dataset),))
        else:
            feat_log, logit_log, label_log = np.load(save_name, allow_pickle=True)
            feat_log, logit_log = feat_log.T, logit_log.T

# ====================== Save features for the OOD data
for out_loader in ood_loaders:

    dset_name = out_loader.dataset.name

    # only sample at most the same amount of data in the ID test set
    if len(out_loader.dataset) > len(id_test_loader.dataset):
        print(f"Too many samples. Will sample {len(id_test_loader.dataset)} out of {len(out_loader.dataset)}")
        num_samples = len(id_test_loader.dataset)
    else:
        num_samples = len(out_loader.dataset)

    # allocate memory
    if args.large_scale:
        ood_feat_log_file = f"{save_folder}/ood_{dset_name}_feat.mmap"
        ood_logit_or_projFeat_log_file = f"{save_folder}/ood_{dset_name}_proj.mmap" if embed_mode \
            else f"{save_folder}/ood_{dset_name}_score.mmap" 
        if os.path.exists(ood_feat_log_file) and not args.rerun:
            print(f"Features of {out_loader.dataset.name} already exist. Going to the next...")
            continue
        else:
            ood_feat_log = np.memmap(ood_feat_log_file, dtype='float32', mode='w+', shape=(num_samples, featdims[-1]))
            ood_logit_or_projFeat_log = np.memmap(ood_logit_or_projFeat_log_file, dtype='float32', mode='w+', shape=(num_samples, proj_dim)) if embed_mode \
                else np.memmap(ood_logit_or_projFeat_log_file, dtype='float32', mode='w+', shape=(num_samples, args.num_classes)) 
    else:
        save_name = f"{save_folder}/ood_{dset_name}_alllayers.npy"
        if os.path.exists(save_name) and not args.rerun:
            print(f"Features of {out_loader.dataset.name} already exist. Going to the next...")
            continue
        else:
            ood_feat_log = np.zeros((num_samples, sum(featdims)))
            ood_logit_log = np.zeros((num_samples, args.num_classes))

    # run
    print(f"\n Extracting the features of {out_loader.dataset.name}...")

    # get start
    net.eval()
    with torch.no_grad():
        with tqdm(total=num_samples) as pbar:
            for batch_idx, (inputs, _) in enumerate(out_loader):
                # if batch_idx >= 1:
                    # break
                inputs = inputs.to(device)
                start_ind = batch_idx * args.test_bs 
                end_ind = min((batch_idx + 1) * args.test_bs, num_samples)

                # NOTE: by default the supcon model WON'T normalize feature
                logit_or_projFeat, feature_list = net.feature_list(inputs)

                # save all features for small scale, but last feature for large scale
                if args.large_scale:
                    feat = F.adaptive_avg_pool2d(feature_list[-1], 1).squeeze()
                else:
                    feat = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1) 

                ood_feat_log[start_ind:end_ind, :] = feat.data.cpu().numpy()[:end_ind - start_ind, :]
                ood_logit_or_projFeat_log[start_ind:end_ind] = logit_or_projFeat.data.cpu().numpy()[:end_ind - start_ind]

                pbar.update(args.test_bs)

                # quit if sampled enough data
                if end_ind >= num_samples:
                    break

        if not args.large_scale:
            np.save(save_name, (ood_feat_log.T, ood_logit_log.T))
