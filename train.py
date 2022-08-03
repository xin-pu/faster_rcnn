from torch import optim
from torch.utils.data import DataLoader

from cfg.plan_config import TrainPlan
from dataset.dataset_generator import ImageDataSet
from main.faster_rcnn import FasterRCNN

net = FasterRCNN()

trainPlan = TrainPlan("cfg/voc_train.yml")

dataset = ImageDataSet(trainPlan)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = optim.NAdam(net.parameters(), lr=0.01)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels_box = data
        inputs = inputs
        labels = labels_box[..., 0]
        bbox = labels[..., 1:]
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        roi_cls_locs, roi_scores, pred_rois, pred_roi_indices = net(inputs, labels, bbox)

        optimizer.step()

        # print statistics

        print(end="\rEpoch: {:05d}\tbatch: {:05d}\tloss: {:>.4f}".format(epoch, i, 1))
        print("\r\n")
