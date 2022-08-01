import torch
from torch import nn
from torch.nn import functional as f

from nets.backbone import get_feature_extractor_classifier
from utils.anchor import generate_anchor_base


# 候选框提取网络
# forward预测候选框
# 在输入的特征图上，构造了一个分类分支和一个坐标回归分支
class RegionProposalNetwork(nn.Module):

    def __init__(self,
                 in_channels=512,
                 mid_channels=512,
                 ratios=(0.5, 1, 2),
                 anchor_scales=(8, 16, 32),
                 feat_stride=16):
        super(RegionProposalNetwork, self).__init__()

        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.n_anchor = n_anchor = self.anchor_base.shape[0]

        self.conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.location_regression_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)  # 卷积层=》坐标
        self.confidence_classify_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)  # 卷积层=》正负样本分类

        # 初始化各层参数
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = f.relu(self.conv(x))
        pred_cls_scores = self.confidence_classify_layer(x)  # [B,50*50*9,2]
        pred_locations = torch.sigmoid(self.location_regression_layer(x))  # [B,50,50*9,4]
        # Todo Sigmoid or Softmax ?

        batch_size, _, height, width = x.shape

        pred_locations = pred_locations.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # [B,50*50*9,4]

        pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()  # [B,50,50,9,2]

        objectness_score = pred_cls_scores.view(batch_size, height, width, -1, 2)[:, :, :, :, 1] \
            .contiguous().view(batch_size, -1)
        pred_cls_scores = pred_cls_scores.view(batch_size, -1, 2)
        return pred_cls_scores, pred_locations, objectness_score


if __name__ == "__main__":
    image = torch.Tensor(2, 3, 800, 800)
    # [22500,4] = [50*50*9,4]
    fe = get_feature_extractor_classifier()
    feature = fe(image)

    rpn = RegionProposalNetwork(512, 512)
    rpn_cls, rpn_loc, rpn_obj = rpn(feature)
    print("rpn_cls:{}\r\nrpn_loc:{}\r\nrpn_obj:{}".format(rpn_cls.shape, rpn_loc.shape, rpn_obj.shape))
