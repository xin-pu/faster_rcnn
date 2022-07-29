import numpy as np
import torch
import torchvision.ops
from numpy import ndarray

from nets.backbone import get_feature_extractor
from nets.region_proposal_network import RegionProposalNetwork, enumerate_shifted_anchor


def cvt_location_to_bbox(pred_locations: ndarray, anchor: ndarray):
    """
    通过归一化处理后中间变量的准确值（就是模型预测的输出值）Pred Locations，与对应的候选框 Anchor
    计算得到预测的实际框 [y1,x1,y2,x2]形式
    x = (w_{a} * ctr_x_{p}) + ctr_x_{a}
    y = (h_{a} * ctr_x_{p}) + ctr_x_{a}
    h = np.exp(h_{p}) * h_{a}
    w = np.exp(w_{p}) * w_{a}
    :param anchor: 原始候选框 [y1,x1,y2,x2]
    :param pred_locations: [ty,tx,th,tw] 归一化处理之后中间变量的准确值
    :return: [y1,x1,y2,x2]
    """
    if anchor.shape[0] == 0:
        return np.zeros((0, 4), dtype=pred_locations.dtype)

    # 转换anchor格式从y1, x1, y2, x2 到ctr_x, ctr_y, h, w ：
    anchor = anchor.astype(anchor.dtype, copy=False)
    ph = anchor[:, 2] - anchor[:, 0]
    pw = anchor[:, 3] - anchor[:, 1]
    px = anchor[:, 0] + 0.5 * ph
    py = anchor[:, 1] + 0.5 * pw

    # 转换预测locs
    ty = pred_locations[:, 0::4]
    tx = pred_locations[:, 1::4]
    th = pred_locations[:, 2::4]
    tw = pred_locations[:, 3::4]

    gy = ty * ph[:, np.newaxis] + px[:, np.newaxis]
    gx = tx * pw[:, np.newaxis] + py[:, np.newaxis]
    gh = np.exp(th) * ph[:, np.newaxis]
    gw = np.exp(tw) * pw[:, np.newaxis]

    # 转换 [ctr_x, ctr_y, h, w]为[y1, x1, y2, x2]格式：
    g_bbox = np.zeros(pred_locations.shape, dtype=pred_locations.dtype)
    g_bbox[:, 0::4] = gy - 0.5 * gh
    g_bbox[:, 1::4] = gx - 0.5 * gw
    g_bbox[:, 2::4] = gy + 0.5 * gh
    g_bbox[:, 3::4] = gx + 0.5 * gw

    return g_bbox


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

    def call(self, pred_locations, objectness_score, anchor, img_size, scale=1.):
        """
        -- 将回归值恢复到原图检测框
        -- 尺寸裁剪至图像尺寸内
        1. 先按min_size_threshold抑制不符合的候选框
        2. 按pre_nms数量抑制
        3. 按重合度nms_iou_thresh，非极大值抑制
        4. 按post_nms数量抑制
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
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        # 先按 min_size_threshold 抑制不符合的候选框
        min_size = self.min_size_threshold * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = objectness_score[keep]

        # 按分数排序，取前n_pre_nms个
        order = score.ravel().argsort()[::-1]
        order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # 非极大值抑制 根据Score >0.7 进行非极大值抑制
        keep = torchvision.ops.nms(torch.from_numpy(roi).cuda(),
                                   torch.from_numpy(score).cuda(),
                                   self.nms_iou_thresh)
        keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        return roi


if __name__ == "__main__":
    image = torch.Tensor(2, 3, 800, 800)
    # [22500,4] = [50*50*9,4]
    fe = get_feature_extractor()
    feature = fe(image)

    rpn = RegionProposalNetwork(512, 512)
    rpn_cls, rpn_loc, rpn_obj = rpn(feature)
    print("rpn_cls:{}\r\nrpn_loc:{}\r\nrpn_obj:{}".format(rpn_cls.shape, rpn_loc.shape, rpn_obj.shape))

    anchors = enumerate_shifted_anchor(rpn.anchor_base, 16, 50, 50)
    p = ProposalCreator()
    roi_after_filter = p.call(rpn_loc[0].data.numpy(), rpn_obj[1].data.numpy(), anchors, (800, 800))
    print(roi_after_filter.shape)
