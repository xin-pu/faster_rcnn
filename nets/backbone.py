import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
import numpy as np


def get_feature_extractor_classifier():
    """
    返回VGG16的特征提取层 和 分类层
    :return:
    """
    # the 30th layer of features is relu of conv5_3
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    # noinspection PyTypeChecker
    feature_lay = list(model.features)[:30]

    # freeze top4 conv
    for layer in feature_lay[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    feature_extractor = nn.Sequential(*feature_lay)

    classifier = list(model.classifier)
    del classifier[6]
    del classifier[5]
    del classifier[2]
    classifier = nn.Sequential(*classifier)

    return feature_extractor, classifier


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


if __name__ == "__main__":
    # feature shape should be [batch, 512, M, N]
    # When Image size is [800,800] => M=50,N=50
    # So RPN Input Unit is 512
    fe_extractor, cls = get_feature_extractor_classifier()
    images = torch.Tensor(1, 3, 800, 800)
    features = fe_extractor(images)
    print(features)

    # anchors: [9,4]
    anchors = generate_anchor_base(base_size=16)
    print(anchors.shape)
