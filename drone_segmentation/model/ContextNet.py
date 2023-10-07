from transformers.models.upernet.modeling_upernet import UperNetHead

from transformers import UperNetConfig, UperNetForSemanticSegmentation
import torch.nn as nn
import torch
import timm
import easydict
import math
def freeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = False
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class ContextNet(nn.Module):
    def __init__(self, backbone_name, context_backbone_name, upernet_cfg, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        self.context_backbone = timm.create_model(context_backbone_name, pretrained=pretrained, num_classes=0)
        freeze_model_params(self.context_backbone)
        features_dim = [x['num_chs'] for x in self.backbone.feature_info[-4:]]
        self.head = UperNetHead(upernet_cfg, features_dim)
        self.projector = nn.Conv2d(self.context_backbone.num_features+features_dim[-1], features_dim[-1], 1, 1, 0)
    
    def forward(self, x, context):
        features = self.backbone(x)[-4:]
        last_h, last_w = features[-1].shape[2:]
        context = self.context_backbone.forward_features(context)[:,1:,:]
        c_n, c_l, c_d = context.shape
        c_w = int(math.sqrt(c_l))
        context = context.view(c_n, c_d, c_w, c_w)
        context = nn.functional.adaptive_avg_pool2d(context, (last_h, last_w))
        features[-1] = torch.concat([context, features[-1]], dim=1)
        features[-1] = self.projector(features[-1])
        output = self.head(features)
        return output
    
    
if __name__ == '__main__':
    x = torch.zeros((2,3,512,512))
    x_context = torch.zeros((2,3,518,518))
    upernet_cfg = {"pool_scales": [1,2,3,6], 'num_labels': 1, 'hidden_size': 512}
    upernet_cfg = easydict.EasyDict(upernet_cfg)
    model = ContextNet('timm/efficientnet_b3.ra2_in1k', 'vit_small_patch14_dinov2.lvd142m', upernet_cfg)
    print_trainable_parameters(model)
    out = model(x, x_context)
    print(out.shape)
    