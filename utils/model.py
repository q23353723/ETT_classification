import imp
import timm
import torch.nn as nn
import segmentation_models_pytorch as smp
from utils.utils import patch_first_conv

class SegModel(nn.Module):
    def __init__(self, backbone):
        super(SegModel, self).__init__()
        self.seg = smp.UnetPlusPlus(encoder_name=backbone, encoder_weights='imagenet', classes=1, activation='sigmoid')
        patch_first_conv(self.seg.encoder, 1, 3, True)
    def forward(self,x):
        global_features = self.seg.encoder(x)
        seg_features = self.seg.decoder(*global_features)
        seg_features = self.seg.segmentation_head(seg_features)
        return seg_features

class ClsModel(nn.Module):
    def __init__(self, enet_type, out_dim):
        super(ClsModel, self).__init__()
        self.enet = timm.create_model(enet_type, True)
        self.dropout = nn.Dropout(0.5)
        self.enet.conv_stem.weight = nn.Parameter(self.enet.conv_stem.weight.repeat(1,2//3+1,1,1)[:, :2])
        self.myfc = nn.Linear(self.enet.classifier.in_features, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        h = self.myfc(self.dropout(x))
        return h