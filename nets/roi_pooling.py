from torch import nn
from torchvision.ops import RoIPool
import torch

from targets.anchor_creator import AnchorCreator


# Fixed 对候选框区域的特征图为输入，预测目标框的类别概率和坐标
class VGG16RoIHead(nn.Module):
    """
    目的是执行从不均匀大小到 固定大小的特征地图（feature maps） (例如 7×7)的输入的最大范围池。
    这一层有两个输入
    一个从有几个卷积和最大池（max-pooling）层的深度卷积网络获得的固定大小的特征地图 feature map
    一个 Nx5 矩阵代表一列兴趣区域（regions of interest），N 表示RoIs的个数. 第一列表示影像的索引，剩下的四个是范围的上左和下右的坐标
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self,
                 classifier,
                 n_class,
                 roi_size,
                 spatial_scale,
                 pre_train):
        """

        :param classifier: 从VGG16中获取的线性层
        :param n_class: 包含背景的分类数目
        :param roi_size: ROI后的特征度宽高
        :param spatial_scale: ROI调整比例
        """
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        if not pre_train:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.normal_(m.weight, 0, 0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, rois, roi_indices):

        rois = rois.float()
        roi_indices = roi_indices.float()

        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)

        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool_resize = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool_resize)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)  # Todo 如何定义损失 是否要softmax
        return roi_cls_locs, roi_scores


if __name__ == "__main__":
    from nets.backbone import get_feature_extractor_classifier
    from nets.region_proposal_network import RegionProposalNetwork

    image = torch.Tensor(2, 3, 800, 800).cuda()
    fe_extractor, cls = get_feature_extractor_classifier()
    fe_extractor = fe_extractor.cuda()
    cls = cls.cuda()
    rpn = RegionProposalNetwork(512, 512).cuda()
    head = VGG16RoIHead(classifier=cls, n_class=21, roi_size=7, spatial_scale=16).cuda()
    anchor = AnchorCreator()()
    fe = fe_extractor(image)
    pred_scores_, pred_locs_, pred_rois_, pred_roi_indices_ = rpn(fe, image.shape[2:], anchor)
    print(pred_scores_.shape)
    print(pred_locs_.shape)
    print(pred_rois_)
    print(pred_roi_indices_)
    roi_cls_locs_, roi_scores_ = head(fe, pred_rois_, pred_roi_indices_)
    print(roi_cls_locs_)
    print(roi_scores_)
