import torch
from torch import Tensor


def cvt_location_to_bbox(pred_locations: Tensor, anchor_bbox: Tensor) -> Tensor:
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
        return torch.zeros((0, 4), dtype=pred_locations.dtype)

    # 转换anchor格式从[y1, x1, y2, x2] 到 [ctr_x, ctr_y, h, w] ：
    anchor_height = anchor_bbox[..., 2] - anchor_bbox[..., 0]
    anchor_width = anchor_bbox[..., 3] - anchor_bbox[..., 1]
    anchor_center_x = anchor_bbox[..., 0] + 0.5 * anchor_height
    anchor_center_y = anchor_bbox[..., 1] + 0.5 * anchor_width

    # 转换预测locs
    ty = pred_locations[..., 0::4]
    tx = pred_locations[..., 1::4]
    th = pred_locations[..., 2::4]
    tw = pred_locations[..., 3::4]

    pred_y = ty * anchor_height.unsqueeze(-1) + anchor_center_x.unsqueeze(-1)
    pred_x = tx * anchor_width.unsqueeze(-1) + anchor_center_y.unsqueeze(-1)
    pred_h = torch.exp(th) * anchor_height.unsqueeze(-1)
    pred_w = torch.exp(tw) * anchor_width.unsqueeze(-1)

    # 转换 [ctr_x, ctr_y, h, w]为[y1, x1, y2, x2]格式：
    pred_bbox = torch.zeros(pred_locations.shape, dtype=pred_locations.dtype)
    pred_bbox[..., 0::4] = pred_y - 0.5 * pred_h
    pred_bbox[..., 1::4] = pred_x - 0.5 * pred_w
    pred_bbox[..., 2::4] = pred_y + 0.5 * pred_h
    pred_bbox[..., 3::4] = pred_x + 0.5 * pred_w

    return pred_bbox


def cvt_bbox_to_location(anchor: Tensor, dst_bbox: Tensor) -> Tensor:
    anchor_height = anchor[..., 2] - anchor[..., 0]
    anchor_width = anchor[..., 3] - anchor[..., 1]
    anchor_center_x = anchor[..., 0] + 0.5 * anchor_height
    anchor_center_y = anchor[..., 1] + 0.5 * anchor_width

    base_height = dst_bbox[..., 2] - dst_bbox[..., 0]
    base_width = dst_bbox[..., 3] - dst_bbox[..., 1]
    base_ctr_y = dst_bbox[..., 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[..., 1] + 0.5 * base_width

    eps_value = torch.finfo(anchor_height.dtype).eps
    eps_tensor = torch.empty_like(anchor_height).fill_(eps_value)
    anchor_height = torch.maximum(anchor_height, eps_tensor)
    anchor_width = torch.maximum(anchor_width, eps_tensor)

    dy = (base_ctr_y - anchor_center_x) / anchor_height
    dx = (base_ctr_x - anchor_center_y) / anchor_width
    dh = torch.log(base_height / anchor_height)
    dw = torch.log(base_width / anchor_width)

    location = torch.vstack((dy, dx, dh, dw)).transpose(-1, -2)
    return location


def bbox_iou(bbox_a: Tensor, bbox_b: Tensor) -> Tensor:
    if bbox_a.shape[-1] != 4 or bbox_b.shape[-1] != 4:
        raise IndexError

    # top left
    tl = torch.maximum(bbox_a[..., None, :2], bbox_b[..., :2])
    # bottom right
    br = torch.minimum(bbox_a[..., None, 2:], bbox_b[..., 2:])

    area_a = torch.prod(bbox_a[..., 2:] - bbox_a[..., :2], dim=-1)
    area_b = torch.prod(bbox_b[..., 2:] - bbox_b[..., :2], dim=-1)

    area_i = torch.prod(br - tl, dim=-1) * (tl < br).all(axis=-1)
    return area_i / (area_a[..., None] + area_b - area_i)


if __name__ == "__main__":
    test_anchor = torch.asarray((0., 0., 1., 1., 0., 0., 1., 1.,)).reshape((2, 4))
    test_location = torch.asarray((0.1, 0.1, 0.5, 0.5, .1, 0.1, 0.5, 0.5,)).reshape((2, 4))

    bbox = cvt_location_to_bbox(test_location, test_anchor)
    loc = cvt_bbox_to_location(test_anchor, bbox)
    print(bbox)
    print(loc)

    iou = bbox_iou(test_anchor, bbox)
    print(iou)
