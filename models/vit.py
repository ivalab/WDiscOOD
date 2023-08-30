from pytorch_pretrained_vit import ViT as ViT_Orig
import torch

class ViT(ViT_Orig):
    def feature_list(self, x):
        # NOTE: for now just add the last layer
        logits, feats = self.forward(x)
        B, D = feats.shape
        out_list = [feats.view(B, D, 1, 1)]

        return logits, out_list

    def forward(self, x, layer_index=None):
        if layer_index is not None:
            raise NotImplementedError
        else:
            b, c, fh, fw = x.shape
            x = self.patch_embedding(x)  # b,d,gh,gw
            x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
            if hasattr(self, 'class_token'):
                x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
            if hasattr(self, 'positional_embedding'): 
                x = self.positional_embedding(x)  # b,gh*gw+1,d 
            x = self.transformer(x)  # b,gh*gw+1,d
            if hasattr(self, 'pre_logits'):     # This does not exist for ViT-B-16
                x = self.pre_logits(x)
                x = torch.tanh(x)
            if hasattr(self, 'fc'):
                feats = self.norm(x)[:, 0]  # b,d
                logits = self.fc(feats)  # b,num_classes
            return logits, feats



 