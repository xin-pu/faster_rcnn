import torch
from torch import Tensor, nn
import torch.nn.functional as f

from utils.to_tensor import cvt_tensor


def fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = cvt_tensor(torch.zeros(gt_loc.shape))

    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    loc_loss = smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)

    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss


def smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


class FinalLoss(torch.nn.Module):
    def __init__(self, roi_sigma=10, rpn_sigma=10):
        super(FinalLoss, self).__init__()
        self.roi_sigma = roi_sigma
        self.rpn_sigma = rpn_sigma

    def forward(self,
                rpn_score: Tensor,
                rpn_loc: Tensor,
                roi_cls_score: Tensor,
                roi_cls_loc: Tensor,
                gt_rpn_score: Tensor,
                gt_rpn_loc: Tensor,
                gt_roi_labels: Tensor,
                gt_roi_locs: Tensor):
        # ------------------ RPN losses -------------------#
        rpn_loc_loss = fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_score, self.rpn_sigma)

        rpn_cls_loss = f.cross_entropy(rpn_score, gt_rpn_score, ignore_index=-1)

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        # KeyPoint 只选取Label正确的Box计算损失更新前错误
        roi_loc = roi_cls_loc[cvt_tensor(torch.arange(0, n_sample)).long(), gt_roi_labels]
        roi_loc_loss = fast_rcnn_loc_loss(roi_loc, gt_roi_locs, gt_roi_labels, self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_cls_score, gt_roi_labels)

        return sum([rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss])
