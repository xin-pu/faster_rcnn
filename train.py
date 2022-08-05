from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
import time

from cfg.plan_config import TrainPlan
from dataset.dataset_generator import ImageDataSet
from loss.final_loss import FinalLoss
from main.faster_rcnn import FasterRCNN
from targets.anchor_target_creator import AnchorTargetCreator
from utils.anchor import generate_anchor_base, enumerate_shifted_anchor
from utils.to_tensor import cvt_module


class Train(object):
    def __init__(self, train_plan):
        self.train_plan = train_plan

    def __call__(self, *args, **kwargs):
        train_plan = self.train_plan

        dataloader = self.get_dataloader()
        net = self.get_model()
        loss_net = cvt_module(FinalLoss())
        optimizer = optim.NAdam(net.parameters(), lr=train_plan.learning_rate)

        anchor = self.get_anchor_base()
        anchor_target_creator = AnchorTargetCreator()

        data_batch_epoch = dataloader.__len__()
        epoch = train_plan.epoch
        image_size = (train_plan.input_size, train_plan.input_size)

        for epoch in range(epoch):  # loop over the dataset multiple times
            time_start = time.time()
            running_loss = 0.0

            for i, data in enumerate(dataloader, 0):

                # get the inputs
                inputs, labels_box = data
                batch_size = inputs.shape[0]

                labels = labels_box[..., 0:1]
                bboxes = labels_box[..., 1:]

                bbox_count = len(torch.where(labels[..., -1] >= 0)[0])
                labels = labels.permute(0, 2, 1)[..., 0:bbox_count].permute(0, 2, 1)
                bboxes = bboxes.permute(0, 2, 1)[..., 0:bbox_count].permute(0, 2, 1)

                optimizer.zero_grad()

                # forward
                rpn_scores, rpn_locs, roi_cls_locs, roi_scores, gt_roi_locs, gt_roi_labels = net(inputs, labels, bboxes,
                                                                                                 anchor)
                gt_rpn_loc_c = []
                gt_rpn_label_c = []
                for b in range(batch_size):
                    gt_rpn_loc, gt_rpn_label = anchor_target_creator(bboxes[b], anchor, image_size)
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

                loss.backward()

                optimizer.step()

                e = i + 1
                running_loss += loss.item()
                ave_loss = running_loss / e
                per = 100.0 * e / data_batch_epoch
                cost_time = time.time() - time_start
                rest_time = (data_batch_epoch - e) * cost_time / e

                print(end="\033\rEpoch: {:05d}\tBatch: {:05d}\tLoss: {:>.4f}\tPer:{:>.2f}%\tCost:{:.0f}s\tRest:{:.0f}s"
                      .format(epoch + 1, i, ave_loss, per, cost_time, rest_time))
            torch.save(net.state_dict(), trainPlan.save_file)
            print("\r\n")

    def get_model(self):
        model = cvt_module(FasterRCNN())
        pre_weights = self.train_plan.save_file
        weight_file = Path(pre_weights)
        if weight_file.exists():
            model.load_state_dict(torch.load(pre_weights))
            print("load from {}".format(pre_weights))
        return model

    def get_anchor_base(self):
        base_size, rations, scales = self.train_plan.anchor_base_size, self.train_plan.anchor_ratios, \
                                     self.train_plan.anchor_scales
        grid_height = grid_width = self.train_plan.input_size / base_size
        anchor_base = generate_anchor_base(base_size, rations, scales)
        return enumerate_shifted_anchor(anchor_base, base_size, grid_height, grid_width)

    def get_dataloader(self):
        dataset = ImageDataSet(self.train_plan)
        return DataLoader(dataset, batch_size=self.train_plan.batch_size, shuffle=True)


my_plan = TrainPlan("cfg/voc_train.yml")
trainer = Train(my_plan)
trainer.__call__()
