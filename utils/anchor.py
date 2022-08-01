import numpy as np


def generate_anchor_base(base_size=16,
                         ratios=(0.5, 1, 2),
                         anchor_scales=(8, 16, 32)):
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4))
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


def enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """
    Enumerate all shifted anchors:
    add A anchors (1, A, 4) to
    cell K shifts (K, 1, 4) to get
    shift anchors (K, A, 4)
    reshape to (K*A, 4) shifted anchors
    return (K*A, 4)
    :param anchor_base: 基础Anchor
    :param feat_stride: 特征图每个格子对应的像素 ，如16*16
    :param height: 输入特征的高
    :param width: 输入特征的宽
    :return:所有Anchor [Height*Width*A,4]
    """

    shift_y = np.arange(0, height * feat_stride, feat_stride)  # (0,800,16)
    shift_x = np.arange(0, width * feat_stride, feat_stride)  # (0,800,16)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1)

    a = anchor_base.shape[0]
    k = shift.shape[0]
    anchor = anchor_base.reshape((1, a, 4)) + shift.reshape((1, k, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((k * a, 4))
    return anchor
