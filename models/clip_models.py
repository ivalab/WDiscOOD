import pdb

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import clip


class CLIP_RN50(nn.Module):
    """
    Based on clip model, add features return from each layer.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        self.model, self.trf = clip.load("RN50", device=device)
        self.vm = self.model.visual

    
    def get_transform(self):
        return self.trf


    def forward(self, x):
        """Return vision model feature as well as embed feature
        """
        def stem(x):
            x = self.vm.relu1(self.vm.bn1(self.vm.conv1(x)))
            x = self.vm.relu2(self.vm.bn2(self.vm.conv2(x)))
            x = self.vm.relu3(self.vm.bn3(self.vm.conv3(x)))
            x = self.vm.avgpool(x)
            return x

        x = x.type(self.vm.conv1.weight.dtype)
        x = stem(x)
        x = self.vm.layer1(x)
        x = self.vm.layer2(x)
        x = self.vm.layer3(x)
        feat_map = self.vm.layer4(x)
        embed = self.vm.attnpool(feat_map)

        return embed, F.adaptive_avg_pool2d(feat_map, 1).squeeze()


    def feature_list(self, x):

        def stem(x):
            x = self.vm.relu1(self.vm.bn1(self.vm.conv1(x)))
            x = self.vm.relu2(self.vm.bn2(self.vm.conv2(x)))
            x = self.vm.relu3(self.vm.bn3(self.vm.conv3(x)))
            x = self.vm.avgpool(x)
            return x

        out_list = []
        x = x.type(self.vm.conv1.weight.dtype)
        x = stem(x)
        x = self.vm.layer1(x)
        out_list.append(x)
        x = self.vm.layer2(x)
        out_list.append(x)
        x = self.vm.layer3(x)
        out_list.append(x)
        x = self.vm.layer4(x)
        out_list.append(x)
        embed = self.vm.attnpool(x)

        return embed, out_list


    
if __name__ == "__main__":
    net = CLIP_RN50(device="cuda")
    fake_input = torch.randn((2, 3, 224, 224)).cuda()
    embed, feat = net(fake_input)
    print(feat.shape)
    print(embed.shape)
    embed, feat_list = net.feature_list(fake_input)
    for i in range(len(feat_list)):
        print(f"Shape of {i}th feat map: {feat_list[i].shape}")