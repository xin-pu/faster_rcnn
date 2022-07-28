import numpy as np
from numpy import ndarray


def loc2bbox(pred_locations: ndarray, anchor: ndarray):
    """

    :param anchor:
    :param pred_locations:
    :return: [y1,x1,y2,x2]
    """
    if anchor.shape[0] == 0:
        return np.zeros((0, 4), dtype=pred_locations.dtype)

    anchor = anchor.astype(anchor.dtype, copy=False)

    anchor_height = anchor[:, 2] - anchor[:, 0]
    anchor_width = anchor[:, 3] - anchor[:, 1]
    anchor_center_x = anchor[:, 0] + 0.5 * anchor_height
    anchor_center_y = anchor[:, 1] + 0.5 * anchor_width

    dy = pred_locations[:, 0::4]
    dx = pred_locations[:, 1::4]
    dh = pred_locations[:, 2::4]
    dw = pred_locations[:, 3::4]

    ctr_y = dy * anchor_height[:, np.newaxis] + anchor_center_x[:, np.newaxis]
    ctr_x = dx * anchor_width[:, np.newaxis] + anchor_center_y[:, np.newaxis]
    h = np.exp(dh) * anchor_height[:, np.newaxis]
    w = np.exp(dw) * anchor_width[:, np.newaxis]

    dst_bbox = np.zeros(pred_locations.shape, dtype=pred_locations.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


if __name__ == "__main__":
    anc = np.random.random(size=(1, 4))
    loc = np.zeros(shape=(1, 4))
    d = loc2bbox(loc, anc)
    print(anc)
    print(d)
