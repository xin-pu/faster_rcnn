import torch
from torch import nn
from torch.nn import functional as f
import numpy as np

from nets.backbone import generate_anchor_base, get_feature_extractor


def enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """
    Enumerate all shifted anchors:
    add A anchors (1, A, 4) to
    cell K shifts (K, 1, 4) to get
    shift anchors (K, A, 4)
    reshape to (K*A, 4) shifted anchors
    return (K*A, 4)
    :param anchor_base: 基础Anchor
    :param feat_stride: 特征图每个格子对应的像素 ，如16*16
    :param height: 输入特征的高
    :param width: 输入特征的宽
    :return:所有Anchor [Height*Width*A,4]
    """

    shift_y = np.arange(0, height * feat_stride, feat_stride)  # (0,800,16)
    shift_x = np.arange(0, width * feat_stride, feat_stride)  # (0,800,16)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1)

    a = anchor_base.shape[0]
    k = shift.shape[0]
    anchor = anchor_base.reshape((1, a, 4)) + shift.reshape((1, k, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((k * a, 4))
    return anchor


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
        # self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
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

    def forward(self, out_feature):
        # 模型前传
        x = f.relu(self.conv(out_feature))  # Todo Relu is necessary ?
        pred_cls_scores = self.confidence_classify_layer(x)  # [B,50*50*9,2]
        pred_anchor_locs = self.location_regression_layer(x)  # [B,50,50*9,4]

        batch_size, _, height, width = out_feature.shape

        pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # [B,50*50*9,4]

        pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()  # [B,50,50,9,2]

        objectness_score = pred_cls_scores.view(batch_size, height, width, -1, 2)[:, :, :, :, 1] \
            .contiguous().view(1, -1)
        pred_cls_scores = pred_cls_scores.view(batch_size, -1, 2)
        return pred_cls_scores, pred_anchor_locs, objectness_score


if __name__ == "__main__":
    image = torch.Tensor(1, 3, 800, 800)
    # [22500,4] = [50*50*9,4]
    fe = get_feature_extractor()
    feature = fe(image)

    rpn = RegionProposalNetwork(512, 512)
    rpn_cls, rpn_loc, rpn_obj = rpn(feature)
    print("rpn_cls:{}\r\nrpn_loc:{}\r\nrpn_obj:{}".format(rpn_cls.shape, rpn_loc.shape, rpn_obj.shape))
