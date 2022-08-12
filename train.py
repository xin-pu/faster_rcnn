from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
import time

from cfg.plan_config import TrainPlan
from dataset.dataset_generator import ImageDataSet
from loss.final_loss_v2 import FinalLoss
from main.faster_rcnn import FasterRCNN
from targets.anchor_creator import AnchorCreator
from targets.anchor_target_creator import AnchorTargetCreator
from utils.to_tensor import cvt_module


class Train(object):
    def __init__(self, train_plan):
        self.train_plan = train_plan

    def __call__(self, *args, **kwargs):
        train_plan = self.train_plan

        dataloader = self.get_dataloader()
        net = self.get_model()
        loss_net = cvt_module(FinalLoss(train_plan.rpn_sigma, train_plan.roi_sigma))

        optimizer = self.get_optimizer(net)

        anchor_creator = AnchorCreator()
        anchor_target_creator = AnchorTargetCreator()

        anchor = anchor_creator(train_plan.anchor_base_size,
                                train_plan.anchor_ratios,
                                train_plan.anchor_scales,
                                train_plan.input_size)
        data_batch_epoch = dataloader.__len__()
        epoch = train_plan.epoch
        image_size = (train_plan.input_size, train_plan.input_size)
        loss_list = []

        for epoch in range(epoch):  # loop over the dataset multiple times
            time_start = time.time()
            running_loss = 0.0
            ave_loss = 0
            l1_loss, l2_loss, l3_loss, l4_loss = [], [], [], []

            for i, data in enumerate(dataloader, 0):

                # get the inputs
                inputs, labels_box = data
                batch_size = inputs.shape[0]
                # Fixed Labels convert to long before cal transfer
                labels = labels_box[..., 0:1].long()
                bboxes = labels_box[..., 1:]

                optimizer.zero_grad()

                # forward
                rpn_scores, rpn_locs, roi_cls_locs, roi_scores, gt_roi_locs, gt_roi_labels = net(inputs, labels, bboxes,
                                                                                                 anchor)
                gt_rpn_loc_c = []
                gt_rpn_label_c = []
                for b in range(batch_size):
                    bbox_count = len(torch.where(labels[b, ..., -1] >= 0)[0])
                    batch_bbox = bboxes[b, 0:bbox_count, ...]
                    gt_rpn_loc, gt_rpn_label = anchor_target_creator(anchor, batch_bbox, image_size)

                    gt_rpn_loc_c.append(gt_rpn_loc.unsqueeze(0))
                    gt_rpn_label_c.append(gt_rpn_label.unsqueeze(0))
                gt_rpn_label = torch.concat(gt_rpn_label_c, dim=0)
                gt_rpn_loc = torch.concat(gt_rpn_loc_c, dim=0)

                loss = loss_net(rpn_scores.view(-1, 2),
                                rpn_locs.view(-1, 4),
                                roi_scores,
                                roi_cls_locs,
                                gt_rpn_label.view(-1),
                                gt_rpn_loc.view(-1, 4),
                                gt_roi_labels.view(-1),
                                gt_roi_locs)

                sum_loss = sum(loss)
                # Keypoint 反向传播异常侦测
                sum_loss.backward()

                optimizer.step()

                e = i + 1
                current_loss = sum_loss.item()
                running_loss += current_loss
                ave_loss = running_loss / e
                per = 100.0 * e / data_batch_epoch
                cost_time = time.time() - time_start
                rest_time = (data_batch_epoch - e) * cost_time / e

                l1_loss.append(loss[0].item())
                l2_loss.append(loss[1].item())
                l3_loss.append(loss[2].item())
                l4_loss.append(loss[3].item())
                print(
                    end="\033\rEpoch: {:05d}\tBatch: {:05d}\tLoss: {:>.4f}\t"
                        "Per:{:>.2f}%\tCost:{:.0f}s\tRest:{:.0f}s\t loss {:>.4f} {:>.4f} {:>.4f} {:>.4f}"
                    .format(epoch + 1, i, ave_loss, per, cost_time, rest_time,
                            sum(l1_loss) / (i + 1),
                            sum(l2_loss) / (i + 1),
                            sum(l3_loss) / (i + 1),
                            sum(l4_loss) / (i + 1)))
            if len(loss_list) == 0:
                torch.save(net.state_dict(), self.train_plan.save_file)
                print("\t save weights.")
            if len(loss_list) > 0:
                if ave_loss < min(loss_list):
                    torch.save(net.state_dict(), self.train_plan.save_file)
                    print("\tloss: {:.4f} <= {:.4f}  save weights.".format(ave_loss, min(loss_list)))
            loss_list.append(ave_loss)
            print("\r\n")

    def predict(self, image):
        train_plan = self.train_plan
        anchor_creator = AnchorCreator()

        anchor = anchor_creator(train_plan.anchor_base_size,
                                train_plan.anchor_ratios,
                                train_plan.anchor_scales,
                                train_plan.input_size)

        net = self.get_model(False)
        pred_scores, pred_locs, roi_cls_locs, roi_scores = net.predict(image, anchor)
        return roi_cls_locs, roi_scores

    def get_model(self, train=True):
        feat_stride = self.train_plan.anchor_base_size
        n_fg_class = len(self.train_plan.labels)
        enhance = self.train_plan.enhance
        loc_normalize_mean = self.train_plan.loc_normalize_mean if enhance else [0., 0., 0., 0.]
        loc_normalize_std = self.train_plan.loc_normalize_std if enhance else [0., 0., 0., 0.]

        if train:
            model = cvt_module(FasterRCNN(feat_stride, n_fg_class, loc_normalize_mean, loc_normalize_std,
                                          pre_train=self.train_plan.pre_train))
        else:
            model = cvt_module(FasterRCNN(feat_stride, n_fg_class, loc_normalize_mean, loc_normalize_std,
                                          pre_train=True))
        pre_weights = self.train_plan.save_file
        weight_file = Path(pre_weights)
        if self.train_plan.pre_train and weight_file.exists():
            model.load_state_dict(torch.load(pre_weights))
            print("load from {}".format(pre_weights))
        return model

    def get_optimizer(self, model):
        weight_decay = 0.0005
        lr = self.train_plan.learning_rate
        params = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    # KeyPoint 增加正则项，否则模型会输出NAN，
                    params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
        return optim.Adam(params)

    def get_dataloader(self):
        dataset = ImageDataSet(self.train_plan)
        return DataLoader(dataset, batch_size=self.train_plan.batch_size, shuffle=True)


# Keypoint 正向传播异常侦测
# torch.autograd.set_detect_anomaly(True)

my_plan = TrainPlan("cfg/raccoon_train.yml")
trainer = Train(my_plan)
trainer.__call__()
