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

    def forward(self, bbox, anchor, img_size):
        """
        根据图像输入的bbox,对其Anchor Box定位
        :param bbox: Bound Box坐标，【R,4】 R 是 BoundBox 数量
        :param anchor: Anchor Box 坐标 【S,4】 S 是 Anchor 数量
        :param img_size: 图像尺寸
        :return:
        """
        img_h, img_w = img_size
        n_anchor = len(anchor)

        inside_index = self.get_inside_index(anchor, img_h, img_w)
        anchor = anchor[inside_index]
        argmax_ious, label = self.create_label(inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc = cvt_bbox_to_location(anchor, bbox[argmax_ious])

        # map up to original set of anchors
        label = self.unmap(label, n_anchor, inside_index, fill=-1)
        loc = self.unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def create_label(self, inside_index, anchor, bbox):
        argmax_ious, max_ious, gt_argmax_ious = self.calc_ious(anchor, bbox, inside_index)

        # 初始默认为忽略 -1
        label = to_device(torch.zeros((len(inside_index),))).long() - 1
        # 分配负标签（0）给max_iou小于负阈值[c]的所有anchor boxes：
        label[max_ious < self.neg_iou_thresh] = 0
        # 分配正标签（1）给与ground-truth box[a]的IoU重叠最大的anchor boxes：
        label[gt_argmax_ious] = 1
        # 分配正标签（1）给max_iou大于positive阈值[b]的anchor boxes：
        label[max_ious >= self.pos_iou_thresh] = 1

        # 如果正样本过多，随机采样正样本，
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = torch.where(label == 1)[0]
        if len(pos_index) > n_pos:
            indices = torch.randperm(len(pos_index))[:(len(pos_index) - n_pos)]
            disable_index = pos_index[indices]
            label[disable_index] = -1

        # 如果负样本过多，随机采样负样本，
        n_neg = self.n_sample - torch.sum(label == 1)
        neg_index = torch.where(label == 0)[0]
        if len(neg_index) > n_neg:
            indices = torch.randperm(len(neg_index))[:(len(neg_index) - n_neg)]
            disable_index = neg_index[indices]
            label[disable_index] = -1

        return argmax_ious, label

    @staticmethod
    def calc_ious(anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(dim=1)
        max_ious = ious[torch.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(dim=0)
        gt_max_ious = ious[gt_argmax_ious, torch.arange(ious.shape[1])]
        gt_argmax_ious = torch.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious

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

    @staticmethod
    def unmap(data, count, index, fill=0):
        # Unmap a subset of item (data) back to the original set of items (of
        # size count)

        if len(data.shape) == 1:
            ret = to_device(torch.full((count,), fill, dtype=data.dtype))
            ret[index] = data
        else:
            ret = to_device(torch.full((count,) + data.shape[1:], fill, dtype=data.dtype))
            ret[index, :] = data
        return ret


if __name__ == "__main__":
    from utils.anchor import generate_anchor_base, enumerate_shifted_anchor

    test_bbox = to_device(torch.asarray([[20, 30, 400, 500], [300, 400, 500, 600]])).float()  # [y1, x1, y2, x2] format

    ang_base = generate_anchor_base()
    ang_pattern = enumerate_shifted_anchor(ang_base, 16, 50, 50)

    at = AnchorTargetCreator()
    locs, labs = at(test_bbox, ang_pattern, (800, 800))
    print(locs)
    print(labs)
