import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor


class ROILoss(torch.nn.Module):
    def __init__(self, roi_lambda=10):
        super(ROILoss, self).__init__()
        self.roi_lambda = roi_lambda

    def forward(self,
                roi_cls_score: Tensor,
                roi_cls_loc: Tensor,
                gt_roi_labels: Tensor,
                gt_roi_locs: Tensor):
        gt_roi_label = gt_roi_labels.long()
        cls_loss = f.cross_entropy(roi_cls_score, gt_roi_label)

        n_sample = roi_cls_loc.shape[0]
        roi_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]  # 我们将只使用带有正标签的边界框

        x = torch.abs(gt_roi_locs - roi_loc)
        rpn_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))
        n_reg = rpn_loc_loss.float().sum()  # ignore gt_label==-1 for rpn_loss
        rpn_loc_loss = rpn_loc_loss.sum() / n_reg

        return cls_loss + self.roi_lambda * rpn_loc_loss


if __name__ == "__main__":
    pred_roi_cls_loc = torch.zeros((128, 84)).float()
    pred_roi_cls_score = torch.ones(128, 21).float()
    gt_roi_locs_ = torch.ones((128, 4)).float()
    gt_roi_labels_ = torch.ones(128).long()

    loss = ROILoss()
    rpn_los = loss(pred_roi_cls_score, pred_roi_cls_loc, gt_roi_labels_, gt_roi_locs_)
    print(rpn_los)
