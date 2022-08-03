import torch
from torch import Tensor
from utils.bbox_tools_torch import bbox_iou, cvt_bbox_to_location


class ProposalTargetCreator(object):

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25,
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5,
                 neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self,
                 roi: Tensor,
                 bbox: Tensor,
                 label: Tensor,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):

        n_bbox, _ = bbox.shape

        roi = torch.concat((roi, bbox), dim=0)
        iou = bbox_iou(roi, bbox)

        pos_roi_per_image = torch.round(torch.tensor(self.n_sample * self.pos_ratio))

        # 找到与每个region proposal具有较高IoU的ground truth，并且找到最大的IoU：
        gt_assignment = iou.argmax(dim=1)
        max_iou = iou.max(dim=1)

        # 为每个proposal分配标签
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = torch.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.shape))
        if pos_index.shape > 0:
            indices = torch.randperm(len(pos_index))[:pos_roi_per_this_image]
            pos_index = pos_index[indices]

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = torch.where((max_iou < self.neg_iou_thresh_hi) &
                                (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.shape))
        if neg_index.shape > 0:
            indices = torch.randperm(len(neg_index))[:neg_roi_per_this_image]
            neg_index = neg_index[indices]

        # 现在我们整合正样本索引和负样本索引，及他们各自的标签和region proposals：
        keep_index = torch.concat(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = cvt_bbox_to_location(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc - torch.asarray(loc_normalize_mean).float()) / torch.asarray(loc_normalize_std).float()

        return sample_roi, gt_roi_loc, gt_roi_label
