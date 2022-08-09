import torch

from utils.bbox_tools_torch import cvt_bbox_to_location, bbox_iou
from utils.to_tensor import to_device


class AnchorTargetCreator(torch.nn.Module):

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7,
                 neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        """
        查看一幅图像中的目标，并将它们分配给包含它们的特定的 AnchorBox
        这将用于 RPN loss 的计算
        与ground-truth-box重叠度最高的Intersection-over-Union (IoU)的anchor
        与ground-truth box 的IoU重叠度大于0.7的anchor
        对所有与ground-truth box的IoU比率小于0.3的anchor标记为负标签
        anchor既不是正样本的也不是负样本，对训练没有帮助

        :param n_sample: 随机采样数量
        :param pos_iou_thresh: 正样本IOU阈值，为下限
        :param neg_iou_thresh: 负样本IOU阈值，为上限
        :param pos_ratio: 正样本比例
        """
        super(AnchorTargetCreator, self).__init__()
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def forward(self, anchor, bbox, img_size):
        """
        对anchor boxes分配标签和位置
        :param bbox: Bound Box坐标，【R,4】 R 是 BoundBox 数量
        :param anchor: Anchor Box 坐标 【S,4】 S 是 Anchor 数量
        :param img_size: 图像尺寸
        :return:
        """
        img_h, img_w = img_size
        n_anchor = len(anchor)

        # 找到有效的anchor boxes的索引，并且生成索引数组，
        inside_index = self.get_inside_index(anchor, img_h, img_w)
        # 生成有效anchor boxes数组：
        anchor = anchor[inside_index]
        # 分配标签数组
        label, argmax_ious = self.create_label(inside_index, anchor, bbox)
        # 为所有有效的anchor box分配anchor locs，而不考虑其标签
        loc = cvt_bbox_to_location(anchor, bbox[argmax_ious])

        # 用inside_index变量将他们映射到原始的anchors，无效的anchor box标签填充-1（忽略），位置填充0
        #
        anchor_labels = to_device(torch.full((n_anchor,), -1, dtype=label.dtype))
        anchor_labels[inside_index] = label

        anchor_locations = to_device(torch.full((n_anchor, loc.shape[1]), 0, dtype=loc.dtype))
        anchor_locations[inside_index, :] = loc

        return anchor_locations, anchor_labels

    def create_label(self, inside_index, anchor, bbox):
        # 计算IOU
        ious = bbox_iou(anchor, bbox)  # [M,N]

        # Case 1 确定每个Ground-truth-Bbox 对应的各自最大IOU的 Anchor-Bbox
        gt_max_ious, gt_argmax_ious = ious.max(dim=0)
        # Case 2 确定每个Anchor-Bbox 对应的各自IOU最大的 Ground-truth-Bbox
        max_ious, argmax_ious = ious.max(dim=1)
        # 确定有max_ious的anchor_boxes（gt_max_ious) 会有多个Anchor BBox与 Ground Truth BBox具有同样的最大的重叠度
        gt_argmax_ious = torch.where(ious == gt_max_ious)[0]

        # 初始默认为忽略 -1
        label = to_device(torch.full((len(inside_index),), -1).long())
        # 分配正标签（1）给与ground-truth box[a]的IoU重叠最大的anchor boxes：
        label[gt_argmax_ious] = 1
        # 分配正标签（1）给max_iou大于positive阈值[b]的anchor boxes：
        label[max_ious >= self.pos_iou_thresh] = 1
        # 分配负标签（0）给max_iou小于负阈值[c]的所有anchor boxes：
        label[max_ious < self.neg_iou_thresh] = 0

        # 如果正样本过多，随机采样正样本，
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = torch.where(label == 1)[0]
        if len(pos_index) > n_pos:
            indices = torch.randperm(len(pos_index))[:(len(pos_index) - n_pos)]
            label[pos_index[indices]] = -1

        # 如果负样本过多，随机采样负样本，
        n_neg = self.n_sample - torch.sum(label == 1)
        neg_index = torch.where(label == 0)[0]
        if len(neg_index) > n_neg:
            indices = torch.randperm(len(neg_index))[:(len(neg_index) - n_neg)]
            label[neg_index[indices]] = -1

        return label, argmax_ious,

    @staticmethod
    def get_inside_index(anchor, h, w):
        """
        到所有有效anchor boxes的索引：
        :param anchor:
        :param h:
        :param w:
        :return:
        """
        index_inside = torch.where(
            (anchor[:, 0] >= 0) &
            (anchor[:, 1] >= 0) &
            (anchor[:, 2] <= h) &
            (anchor[:, 3] <= w))[0]
        return index_inside


if __name__ == "__main__":
    from targets.anchor_creator import AnchorCreator

    test_bbox = to_device(torch.asarray([[20, 30, 400, 500], [300, 400, 500, 600]])).float()  # [y1, x1, y2, x2] format

    ang_pattern = AnchorCreator()()

    anchor_target_creator = AnchorTargetCreator()
    locs, labs = anchor_target_creator(ang_pattern, test_bbox, (800, 800))
    print(locs.shape)
    print(labs.shape)
