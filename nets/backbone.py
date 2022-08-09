import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights

from utils.to_tensor import to_device


def get_feature_extractor_classifier():
    """
    返回VGG16的特征提取层 和 分类层 用于
    特征提取层用于构建猪肝网络
    分类层用于对池化后感兴趣区域分类
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


if __name__ == "__main__":
    # feature shape should be [batch, 512, M, N]
    # When Image size is [800,800] => M=50,N=50
    # So RPN Input Unit is 512
    fe_extractor, cls = get_feature_extractor_classifier()
    print(fe_extractor)
    print(cls)
    fe_extractor = fe_extractor.cuda()
    cls = cls.cuda()
    images = to_device(torch.Tensor(1, 3, 800, 800)).float()
    features = fe_extractor(images)
    print(features)
