import numpy as np
import torch
from numpy import ndarray


def cvt_location_to_bbox(pred_locations: ndarray, anchor_bbox: ndarray):
    """
    通过归一化处理后中间变量的准确值（就是模型预测的输出值）Pred Locations，与对应的候选框 Anchor
    计算得到预测的实际框 [y1,x1,y2,x2]形式
    x = (w_{a} * ctr_x_{p}) + ctr_x_{a}
    y = (h_{a} * ctr_x_{p}) + ctr_x_{a}
    h = np.exp(h_{p}) * h_{a}
    w = np.exp(w_{p}) * w_{a}
    :param pred_locations: [ty,tx,th,tw] 归一化处理之后中间变量的准确值
    :param anchor_bbox: 原始候选框 [y1,x1,y2,x2]
    :return: pred_bbox: 位置修正后的候选框 [y1,x1,y2,x2]
    """
    if anchor_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=pred_locations.dtype)

    # 转换anchor格式从[y1, x1, y2, x2] 到 [ctr_x, ctr_y, h, w] ：
    anchor_height = anchor_bbox[:, 2] - anchor_bbox[:, 0]
    anchor_width = anchor_bbox[:, 3] - anchor_bbox[:, 1]
    anchor_center_x = anchor_bbox[:, 0] + 0.5 * anchor_height
    anchor_center_y = anchor_bbox[:, 1] + 0.5 * anchor_width

    # 转换预测locs
    ty = pred_locations[:, 0::4]
    tx = pred_locations[:, 1::4]
    th = pred_locations[:, 2::4]
    tw = pred_locations[:, 3::4]

    pred_y = ty * anchor_height[:, np.newaxis] + anchor_center_x[:, np.newaxis]
    pred_x = tx * anchor_width[:, np.newaxis] + anchor_center_y[:, np.newaxis]
    pred_h = np.exp(th) * anchor_height[:, np.newaxis]
    pred_w = np.exp(tw) * anchor_width[:, np.newaxis]

    # 转换 [ctr_x, ctr_y, h, w]为[y1, x1, y2, x2]格式：
    pred_bbox = np.zeros(pred_locations.shape, dtype=pred_locations.dtype)
    pred_bbox[:, 0::4] = pred_y - 0.5 * pred_h
    pred_bbox[:, 1::4] = pred_x - 0.5 * pred_w
    pred_bbox[:, 2::4] = pred_y + 0.5 * pred_h
    pred_bbox[:, 3::4] = pred_x + 0.5 * pred_w

    return pred_bbox


def cvt_bbox_to_location(anchor, dst_bbox):
    height = anchor[:, 2] - anchor[:, 0]
    width = anchor[:, 3] - anchor[:, 1]
    ctr_y = anchor[:, 0] + 0.5 * height
    ctr_x = anchor[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    location = np.vstack((dy, dx, dh, dw)).transpose()
    return location


def bbox_iou(bbox_a: ndarray, bbox_b: ndarray) -> ndarray:
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


if __name__ == "__main__":
    test_anchor = np.asarray((0., 0., 1., 1.)).reshape((1, 4))
    test_location = np.asarray((0.1, 0.1, 0.5, 0.5)).reshape((1, 4))
    bbox = cvt_location_to_bbox(test_location, test_anchor)
    loc = cvt_bbox_to_location(test_anchor, bbox)
    print(bbox)
    print(loc)
