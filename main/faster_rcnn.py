import torch
import torch as t
from torch import nn

from nets.backbone import get_feature_extractor_classifier
from nets.region_proposal_network import RegionProposalNetwork
from nets.roi_pooling import VGG16RoIHead
from targets.proposal_target_creator import ProposalTargetCreator
from utils.to_tensor import cvt_tensor


class FasterRCNN(nn.Module):

    def __init__(self,
                 feat_stride=16,
                 n_fg_class=20,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        extractor, classifier = get_feature_extractor_classifier()

        self.feature_extractor = extractor
        self.rpn = RegionProposalNetwork(512,
                                         512,
                                         feat_stride=feat_stride)

        self.head = VGG16RoIHead(n_class=n_fg_class + 1,
                                 roi_size=7,
                                 spatial_scale=(1. / feat_stride),
                                 classifier=classifier)

        self.proposal_target_creator = ProposalTargetCreator()

    def forward(self, x, labels, bbox, anchor, scale=1.):
        img_size = x.shape[2:]
        batch = x.shape[0]

        feature = self.feature_extractor(x)

        pred_scores, pred_locs, pred_rois, pred_roi_indices = self.rpn(feature, img_size, anchor, scale)

        sample_rois = []
        sample_roi_indices = []
        gt_roi_loc_array = []
        gt_roi_label_array = []
        for b in range(batch):
            roi_indices = torch.where(pred_roi_indices == b)[0]
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                pred_rois[roi_indices],
                bbox[b, ...],
                labels[b, ...],
                self.loc_normalize_mean,
                self.loc_normalize_std)
            sample_rois.append(sample_roi)
            i = torch.full((sample_roi.shape[0], 1), b)
            sample_roi_indices.append(cvt_tensor(i.long()))
            gt_roi_loc_array.append(gt_roi_loc)
            gt_roi_label_array.append(gt_roi_label)

        sample_roi_t = torch.concat(sample_rois, dim=0)
        sample_roi_indices_t = torch.concat(sample_roi_indices, dim=0)
        sample_roi_indices_t = sample_roi_indices_t.view(sample_roi_t.shape[0])
        roi_cls_locs, roi_scores = self.head(feature, sample_roi_t, sample_roi_indices_t)
        gt_roi_locs = torch.concat(gt_roi_loc_array, dim=0)
        gt_roi_labels = torch.concat(gt_roi_label_array, dim=0)
        return pred_scores, pred_locs, roi_cls_locs, roi_scores, gt_roi_locs, gt_roi_labels


if __name__ == "__main__":
    f_rcnn = FasterRCNN()

    inputImage = torch.Tensor(1, 3, 800, 800).float()
    print(f_rcnn)
