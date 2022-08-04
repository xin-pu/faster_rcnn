import torch
from torch import optim
from torch.utils.data import DataLoader
import time

from cfg.plan_config import TrainPlan
from dataset.dataset_generator import ImageDataSet
from loss.rpn_loss import RPNLoss
from main.faster_rcnn import FasterRCNN
from targets.anchor_target_creator import AnchorTargetCreator
from utils.anchor import generate_anchor_base, enumerate_shifted_anchor

net = FasterRCNN().cuda()
rpn_loss = RPNLoss().cuda()
anchor_target_creator = AnchorTargetCreator().cuda()
anchor = enumerate_shifted_anchor(generate_anchor_base(), 16, 50, 50).cuda()

trainPlan = TrainPlan("cfg/voc_train.yml")

dataset = ImageDataSet(trainPlan)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
data_batch_epoch = dataloader.__len__()

optimizer = optim.NAdam(net.parameters(), lr=0.001)

for epoch in range(100):  # loop over the dataset multiple times
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
        rpn_scores, rpn_locs = net(inputs, labels, bboxes, anchor)
        gt_rpn_loc_c = []
        gt_rpn_label_c = []
        for b in range(batch_size):
            gt_rpn_loc, gt_rpn_label = anchor_target_creator(bboxes[b], anchor, (800, 800))
            gt_rpn_loc_c.append(gt_rpn_loc.unsqueeze(0))
            gt_rpn_label_c.append(gt_rpn_label.unsqueeze(0))
        gt_rpn_label = torch.concat(gt_rpn_label_c, dim=0)
        gt_rpn_loc = torch.concat(gt_rpn_loc_c, dim=0)

        loss = rpn_loss(rpn_scores.view(-1, 2), rpn_locs.view(-1, 4), gt_rpn_label.view(-1), gt_rpn_loc.view(-1, 4))
        # backward
        loss.backward()

        # optimize
        optimizer.step()

        # print statistics
        e = i + 1
        running_loss += loss.item()
        ave_loss = running_loss / e
        per = 100.0 * e / data_batch_epoch
        cost_time = time.time() - time_start
        rest_time = (data_batch_epoch - e) * cost_time / e

        print(end="\033\rEpoch: {:05d}\tBatch: {:05d}\tLoss: {:>.4f}\tPer:{:>.2f}%\tCost:{:.0f}s\tRest:{:.0f}s"
              .format(epoch + 1, i, ave_loss, per, cost_time, rest_time))
