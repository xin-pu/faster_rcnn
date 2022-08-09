from torch import nn
from torch.nn import functional as f

from targets.proposal_creator import ProposalCreator
import torch


class RegionProposalNetwork(nn.Module):
    """
    候选框提取网络
    forward预测候选框
    在输入的特征图上，构造了一个分类分支和一个坐标回归分支
    """

    def __init__(self,
                 in_channels=512,
                 mid_channels=512,
                 feat_stride=16,
                 base_anchor_length=9):
        super(RegionProposalNetwork, self).__init__()

        self.feat_stride = feat_stride

        self.conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.location_regression_layer = nn.Conv2d(mid_channels, base_anchor_length * 4, 1, 1, 0)  # 卷积层=》坐标
        self.confidence_classify_layer = nn.Conv2d(mid_channels, base_anchor_length * 2, 1, 1, 0)  # 卷积层=》正负样本分类
        self.proposal_layer = ProposalCreator()

        # 初始化各层参数
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, image_size, anchor, scale=1.):

        batch_size, _, height, width = x.shape
        x_relu = f.relu(self.conv(x))
        pred_cls_scores = torch.sigmoid(self.confidence_classify_layer(x_relu))  # [B,50*50*9,2]
        pred_locations = self.location_regression_layer(x_relu)  # [B,50,50*9,4]

        pred_locations = pred_locations.permute(0, 2, 3, 1) \
            .contiguous() \
            .view(batch_size, -1, 4)  # [B,50*50*9,4]

        pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1) \
            .contiguous()  # [B,50,50,9,2]

        objectness_score = pred_cls_scores.view(batch_size, height, width, -1, 2)[:, :, :, :, 1] \
            .contiguous() \
            .view(batch_size, -1)
        pred_cls_scores = pred_cls_scores \
            .view(batch_size, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(batch_size):
            # 根据预测的结果生成 ROIS
            roi = self.proposal_layer(pred_locations[i],
                                      objectness_score[i],
                                      anchor, image_size, scale)
            batch_index = i * torch.ones((len(roi),)).long()
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = torch.concat(rois, dim=0)
        roi_indices = torch.concat(roi_indices, dim=0)
        return pred_cls_scores, pred_locations, rois, roi_indices


if __name__ == "__main__":
    from nets.backbone import get_feature_extractor_classifier
    from targets.anchor_creator import AnchorCreator
    from utils.to_tensor import cvt_module, to_device

    image = to_device(torch.Tensor(1, 3, 800, 800))
    # [22500,4] = [50*50*9,4]
    backbone, _ = get_feature_extractor_classifier()
    backbone = cvt_module(backbone)
    feature = backbone(image)

    ac = AnchorCreator()
    rpn = RegionProposalNetwork(512, 512)
    rpn = cvt_module(rpn)

    anchors = ac()
    pred_scores, pred_locs, pred_rois, pred_roi_indices = rpn(feature, image.shape[2:], anchors)
    print("rpn_cls:{}\r\nrpn_loc:{}".format(pred_scores.shape, pred_locs.shape, pred_rois.shape))
    print("rois:{}\r\nroi_indices:{}".format(pred_rois.shape, pred_roi_indices.shape))
