import torch
from torch import Tensor
import torch.nn.functional as f


class FinalLoss(torch.nn.Module):
    def __init__(self, rpn_lambda=10, roi_lambda=10):
        super(FinalLoss, self).__init__()
        self.roi_lambda = roi_lambda
        self.rpn_lambda = rpn_lambda

    def forward(self,
                rpn_score: Tensor,
                rpn_loc: Tensor,
                roi_cls_score: Tensor,
                roi_cls_loc: Tensor,
                gt_rpn_score: Tensor,
                gt_rpn_loc: Tensor,
                gt_roi_labels: Tensor,
                gt_roi_locs: Tensor):
        rpn_cls_loss = f.cross_entropy(rpn_score, gt_rpn_score, ignore_index=-1)

        pos = gt_rpn_score > 0
        mask = pos.unsqueeze(1).expand_as(rpn_loc)
        mask_loc_preds = rpn_loc[mask].view(-1, 4)
        mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)

        x = torch.abs(mask_loc_targets - mask_loc_preds)
        rpn_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))
        n_reg = (gt_rpn_score > 0).float().sum()  # ignore gt_label==-1 for rpn_loss
        rpn_loc_loss = rpn_loc_loss.sum() / n_reg

        gt_roi_label = gt_roi_labels.long()
        roi_cls_loss = f.cross_entropy(roi_cls_score, gt_roi_label)

        n_sample = roi_cls_loc.shape[0]
        roi_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]  # 我们将只使用带有正标签的边界框

        x = torch.abs(gt_roi_locs - roi_loc)
        roi_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))
        # Fixed bug
        n_reg = (gt_roi_label >= 0).float().sum()
        roi_loc_loss = roi_loc_loss.sum() / n_reg

        return rpn_cls_loss + self.rpn_lambda * rpn_loc_loss + roi_cls_loss + self.roi_lambda * roi_loc_loss
