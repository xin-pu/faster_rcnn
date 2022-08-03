import torch
import torch as t
from torch import nn

from nets.backbone import get_feature_extractor_classifier
from nets.region_proposal_network import RegionProposalNetwork
from nets.roi_pooling import VGG16RoIHead
from targets.proposal_target_creator import ProposalTargetCreator


def no_grad(fun):
    """
    装饰器模式，不执行梯度
    :param fun: 传入方法
    :return: 装饰后方法
    """

    def new_f(*args, **kwargs):
        with t.no_grad():
            return fun(*args, **kwargs)

    return new_f


class FasterRCNN(nn.Module):

    def __init__(self,
                 feat_stride=16,
                 n_fg_class=20,
                 ratios=(0.5, 1, 2),
                 anchor_scales=(8, 16, 32),
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        extractor, classifier = get_feature_extractor_classifier()

        self.feature_extractor = extractor
        self.rpn = RegionProposalNetwork(512, 512,
                                         ratios=ratios,
                                         anchor_scales=anchor_scales,
                                         feat_stride=feat_stride)

        self.head = VGG16RoIHead(n_class=n_fg_class + 1,
                                 roi_size=7,
                                 spatial_scale=(1. / feat_stride),
                                 classifier=classifier)

        self.proposal_target_creator = ProposalTargetCreator()

    def forward(self, x, labels, bbox, scale=1.):
        img_size = x.shape[2:]

        feature = self.feature_extractor(x)

        pred_scores, pred_locs, pred_rois, pred_roi_indices = self.rpn(feature, img_size, scale)

        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            pred_rois,
            bbox,
            labels,
            self.loc_normalize_mean,
            self.loc_normalize_std)

        roi_cls_locs, roi_scores = self.head(feature, sample_roi, pred_roi_indices)

        return roi_cls_locs, roi_scores, pred_rois, pred_roi_indices


if __name__ == "__main__":
    f_rcnn = FasterRCNN()

    inputImage = torch.Tensor(1, 3, 800, 800).float()
    a, b, c, d = f_rcnn(inputImage)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
