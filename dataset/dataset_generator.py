import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *

import cv2

from cfg.plan_config import TrainPlan


class ImageDataSet(Dataset):

    def __init__(self, train_plan):
        """
        构造目标检测的数据集
        :param train_plan
        """
        self.train_plan = train_plan

        self.image_files = pd.read_csv(train_plan.image_index_file, header=None).iloc[:, 0].values
        self.annot_files = self.get_annot_file(self.image_files)
        self.len = self.image_files.__len__()

        self.transform = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def get_annot_file(self, image_files):
        ann = []
        for f in image_files:
            file_name, extension = os.path.splitext(os.path.basename(f))
            ann.append(os.path.join(self.train_plan.annot_encode_folder, "{}.txt".format(file_name)))
        return ann

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_file = self.image_files[index]
        annot_file = self.annot_files[index]
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (800, 800))
        image = image / 255.
        image = image.transpose(2, 0, 1)
        return image, torch.zeros(3)

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            if key == "image_files" or key == "annot_files":
                pass
            else:
                info += "{}:\t{}\r\n".format(key, value)
        return info


if __name__ == "__main__":
    trainPlan = TrainPlan("../cfg/voc_train.yml")
    print(trainPlan)
    dataset = ImageDataSet(trainPlan)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for images, targets in dataloader:
        print(images.shape)
