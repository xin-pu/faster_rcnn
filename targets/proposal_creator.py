import torch
import torchvision.ops

from utils.bbox_tools_torch import cvt_location_to_bbox


class ProposalCreator(object):

    def __init__(self,
                 parent_model=None,
                 min_size_threshold=16,
                 nms_iou_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300):
        """

        :param parent_model:
        :param nms_iou_thresh:
        :param n_train_pre_nms:
        :param n_train_post_nms:
        :param n_test_pre_nms:
        :param n_test_post_nms:
        :param min_size_threshold:
        """
        self.parent_model = parent_model
        self.nms_iou_thresh = nms_iou_thresh

        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms

        self.min_size_threshold = min_size_threshold

    def __call__(self, pred_locations, objectness_score, anchor, img_size, scale=1.):
        """
        -- 将回归值恢复到原图检测框
        -- 尺寸裁剪至图像尺寸内
        1. 先按min_size_threshold抑制不符合的候选框
        2. 按pre_nms数量抑制
        3. 按重合度nms_iou_thresh，post_nms非极大值抑制
        -- 返回抑制后感兴趣框
        :param pred_locations: [Batch,50*50*9,4]
        :param objectness_score:[Batch,50*50*9]
        :param anchor:  shape: [9,4]
        :param img_size:
        :param scale:
        :return:
        """
        if self.parent_model is None or self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # 将回归值恢复到原图检测框
        roi = cvt_location_to_bbox(pred_locations, anchor)

        # 尺寸裁剪至图像尺寸内
        roi[:, slice(0, 4, 2)] = torch.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = torch.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        # 先按 min_size_threshold 抑制不符合的候选框
        min_size = self.min_size_threshold * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = torch.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = objectness_score[keep]

        # 按分数排序，取前n_pre_nms个
        order = score.ravel().argsort()
        order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # 非极大值抑制 根据Score >0.7 取前n_post_nms个进行非极大值抑制
        keep = torchvision.ops.nms(roi,
                                   score,
                                   self.nms_iou_thresh)
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


if __name__ == "__main__":
    pass
