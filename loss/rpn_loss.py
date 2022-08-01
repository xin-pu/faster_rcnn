import torch.nn
import torch.nn.functional as f
from torch import Tensor


class RPNLoss(torch.nn.Module):
    def __init__(self, rpn_lambda=10):
        super(RPNLoss, self).__init__()
        self.rpn_lambda = rpn_lambda

    def forward(self,
                rpn_score: Tensor,
                rpn_loc: Tensor,
                gt_rpn_score: Tensor,
                gt_rpn_loc: Tensor):
        cls_loss = f.cross_entropy(rpn_score, gt_rpn_score.long(), ignore_index=-1)

        pos = gt_rpn_score > 0
        mask = pos.unsqueeze(1).expand_as(rpn_loc)
        mask_loc_preds = rpn_loc[mask].view(-1, 4)
        mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)

        x = torch.abs(mask_loc_targets - mask_loc_preds)
        rpn_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))
        n_reg = (gt_rpn_score > 0).float().sum()
        rpn_loc_loss = rpn_loc_loss.sum() / n_reg

        return cls_loss + self.rpn_lambda * rpn_loc_loss


if __name__ == "__main__":
    pred_anchor_locs_ = torch.Tensor(12321, 4)
    pred_cls_scores_ = torch.Tensor(12321, 2)
    anchor_locations_ = torch.Tensor(12321, 4)
    anchor_labels_ = torch.Tensor(12321)

    loss = RPNLoss()
    rpn_los = loss(pred_cls_scores_, pred_anchor_locs_, anchor_labels_, anchor_locations_)
    print(rpn_los)
