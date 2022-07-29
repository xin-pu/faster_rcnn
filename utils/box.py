import numpy as np
from numpy import ndarray


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


if __name__ == "__main__":
    anc = np.random.random(size=(1, 4))
    loc = np.zeros(shape=(1, 4))
    d = cvt_location_to_bbox(loc, anc)
    print(anc)
    print(d)
