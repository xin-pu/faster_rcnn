import torch
from torch import Tensor
from utils.bbox_tools_torch import bbox_iou, cvt_bbox_to_location
from utils.to_tensor import cvt_tensor


class ProposalTargetCreator(object):

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25,
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5,
                 neg_iou_thresh_lo=0.0):
        """

        :param n_sample:
        :param pos_ratio:
        :param pos_iou_thresh:
        :param neg_iou_thresh_hi:
        :param neg_iou_thresh_lo:
        """
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
        """

        :param roi: Proposal Creator 传递的ROI区域
        :param bbox: Ground Truth Bbox
        :param label: Ground Truth Labels
        :param loc_normalize_mean:
        :param loc_normalize_std:
        :return:
        """
        n_bbox, _ = bbox.shape

        # roi = torch.concat((roi, bbox), dim=0)  # 将Ground Truth 也加入了Sample ROI中
        iou = bbox_iou(roi, bbox)

        # 找到与每个region proposal具有较高IoU的ground truth，并且找到最大的IoU：
        max_iou, gt_assignment = torch.max(iou, dim=1)

        # 为每个proposal分配标签
        # 若未将背景标记为0，则所有的标签 + 1。
        gt_roi_label = label[gt_assignment] + 1

        # 根据每个pos_iou_thresh选择前景rois。希望只保留n_sample*pos_ratio（128*0.25=32）个前景样本，
        # 因此如果只得到少于32个正样本，保持原状。如果得到多余32个前景目标，从中采样32个样本
        pos_index = torch.where(max_iou >= self.pos_iou_thresh)[0]
        pos_count = pos_index.shape[0]
        pos_roi_per_image = torch.round(torch.tensor(self.n_sample * self.pos_ratio))
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_count))
        if pos_count > 0:
            indices = torch.randperm(len(pos_index))[:pos_roi_per_this_image]
            pos_index = pos_index[indices]

        # 针对负[背景]region proposal进行相似处理，如果对于之前分配的ground truth目标，
        # region proposal的IoU在neg_iou_thresh_lo和neg_iou_thresh_hi之间，对该region proposal分配0标签，
        # 从这些负样本中采样n(n_sample-pos_samples,128-32=96)个region proposals。
        neg_index = torch.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_count = neg_index.shape[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_count))
        if neg_count > 0:
            indices = torch.randperm(len(neg_index))[:neg_roi_per_this_image]
            neg_index = neg_index[indices]

        # 现在我们整合正样本索引和负样本索引，及他们各自的标签和region proposals：
        keep_index = torch.concat([pos_index, neg_index])
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = cvt_bbox_to_location(sample_roi, bbox[gt_assignment[keep_index]])
        # Why ?
        # gt_roi_loc = (gt_roi_loc - cvt_tensor(torch.asarray(loc_normalize_mean).float())) / \
        #              cvt_tensor(torch.asarray(loc_normalize_std).float())

        return sample_roi, gt_roi_loc, gt_roi_label


if __name__ == "__main__":
    rois = cvt_tensor(torch.tensor([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3], [0, 0, 4, 4], [0, 0, 4, 5]]).float())
    boxes = cvt_tensor(torch.tensor([[0, 0, 0.9, 1], [0, 0, 2.1, 2]]).float())
    labels = cvt_tensor(torch.tensor([[3], [4]]).float())

    at = ProposalTargetCreator()
    a_, b_, c_ = at(rois, boxes, labels)
    print(a_)
    print(b_)
    print(c_)
