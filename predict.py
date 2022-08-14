from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torchvision.ops import nms

from cfg.plan_config import TrainPlan
from main.faster_rcnn import FasterRCNN
from targets.anchor_creator import AnchorCreator
from utils.to_tensor import cvt_module
from utils.bbox_tools_torch import cvt_location_to_bbox


class Predict(object):
    def __init__(self, train_plan):
        self.train_plan = train_plan

    def __call__(self, image):
        train_plan = self.train_plan

        anchor_creator = AnchorCreator()

        anchor = anchor_creator(train_plan.anchor_base_size,
                                train_plan.anchor_ratios,
                                train_plan.anchor_scales,
                                train_plan.input_size)

        net = self.get_model()
        roi_cls_loc, roi_scores, rois, _ = net.predict(image, anchor)
        return roi_cls_loc, roi_scores, rois

    def get_model(self):
        feat_stride = self.train_plan.anchor_base_size
        n_fg_class = len(self.train_plan.labels)
        enhance = self.train_plan.enhance
        loc_normalize_mean = self.train_plan.loc_normalize_mean if enhance else [0., 0., 0., 0.]
        loc_normalize_std = self.train_plan.loc_normalize_std if enhance else [0., 0., 0., 0.]

        model = cvt_module(FasterRCNN(feat_stride, n_fg_class, loc_normalize_mean, loc_normalize_std,
                                      pre_train=True))

        pre_weights = self.train_plan.save_file
        weight_file = Path(pre_weights)
        if not weight_file.exists():
            raise Exception()

        model.load_state_dict(torch.load(pre_weights))
        print("load from {}".format(pre_weights))
        return model


def get_image(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    height, width, _ = image.shape
    scale_height, scale_width = height / 800., width / 800.

    image = cv2.resize(image, (800, 800))
    image = image / 255.
    image = image.transpose(2, 0, 1)
    image = torch.asarray(image).float()
    return image.cuda(), (scale_height, scale_width)


image_file = r"E:\OneDrive - II-VI Incorporated\Pictures\Saved Pictures\cars.jpg"
test_image, scale = get_image(image_file)
test_image = test_image.unsqueeze(0)

my_plan = TrainPlan("cfg/voc_train.yml")
n_class = len(my_plan.labels) + 1
trainer = Predict(my_plan)

roi_cls_loc, roi_scores, roi = trainer(test_image)
roi_score = roi_scores.data
roi_cls_loc = roi_cls_loc.data

roi_cls_loc = roi_cls_loc.view(-1, n_class, 4)
roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)

cls_bbox = cvt_location_to_bbox(roi_cls_loc, roi)
cls_bbox = cls_bbox.view(-1, n_class * 4)
cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=800)
cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=800)

prob = F.softmax(roi_scores, dim=-1)

bbox = list()
label = list()
score = list()
for la in range(1, n_class):
    cls_bbox_l = cls_bbox.reshape((-1, n_class, 4))[:, la, :]
    prob_l = prob[:, la]
    mask = prob_l > 0.8
    cls_bbox_l = cls_bbox_l[mask]
    prob_l = prob_l[mask]
    keep = nms(cls_bbox_l, prob_l, 0.3)
    bbox.append(cls_bbox_l[keep].detach().cpu().numpy())
    label.append((la - 1) * np.ones((len(keep),)))
    score.append(prob_l[keep].detach().cpu().numpy())

bbox = np.concatenate(bbox, axis=0).astype(np.float32)
label = np.concatenate(label, axis=0).astype(np.int32)
score = np.concatenate(score, axis=0).astype(np.float32)
print(score)
bbox[..., 0] = bbox[..., 0] * scale[0]
bbox[..., 1] = bbox[..., 1] * scale[1]
bbox[..., 2] = bbox[..., 2] * scale[0]
bbox[..., 3] = bbox[..., 3] * scale[1]
bbox = bbox[..., [1, 0, 3, 2]]

image_s = cv2.imread(image_file)

i = 0
for box in bbox:
    min_max = box
    pt1 = (int(min_max[0]), int(min_max[1]))
    pt2 = (int(min_max[2]), int(min_max[3]))
    cv2.rectangle(image_s, pt1, pt2, (255, 255, 0), 1)
    class_name = my_plan.labels[label[i]]
    prob = score[i]

    cv2.putText(image_s, "{0} {1:.2f}%".format(class_name, prob * 100), pt1, cv2.FONT_ITALIC, 1,
                (0, 0, 255), 1,
                lineType=cv2.LINE_AA)
    i += 1

cv2.imshow("Result", image_s)
cv2.waitKey(20000)

