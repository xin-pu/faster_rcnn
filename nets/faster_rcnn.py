import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.models.detection.rpn import RegionProposalNetwork

from nets.vgg16_roi_header import VGG16RoIHead


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
                 extractor,
                 rpn,
                 head,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    def forward(self, x, scale=1.):
        """Forward Faster R-CNN.

        Scaling parameter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class


# 主干网络
class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=(0.5, 1, 2),
                 anchor_scales=(8, 16, 32)):

        extractor, classifier = self.decom_vgg16()

        RegionProposalNetwork
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride)

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier)

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head)

    @staticmethod
    def decom_vgg16():
        # the 30th layer of features is relu of conv5_3
        if opt.caffe_pretrain:
            model = vgg16(pretrained=False)
            if not opt.load_path:
                model.load_state_dict(t.load(opt.caffe_pretrain_path))
        else:
            model = vgg16(not opt.load_path)

        features = list(model.features)[:30]
        classifier = model.classifier

        classifier = list(classifier)
        del classifier[6]
        if not opt.use_drop:
            del classifier[5]
            del classifier[2]
        classifier = nn.Sequential(*classifier)

        # freeze top4 conv
        for layer in features[:10]:
            for p in layer.parameters():
                p.requires_grad = False

        return nn.Sequential(*features), classifier
