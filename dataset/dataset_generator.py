import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *
from utils.to_tensor import *

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
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        height, width, _ = image.shape
        scale_height, scale_width = height / 800., width / 800.
        image = cv2.resize(image, (800, 800))
        image = image / 255.
        image = image.transpose(2, 0, 1)
        image = torch.asarray(image).float()  # opencv 读取维Double 需转float

        annot_file = self.annot_files[index]
        data = pd.read_csv(annot_file, sep=' ', header=None).iloc[:, :].values
        label_bboxes = torch.asarray(data).float().view(-1, 5)

        labels = label_bboxes[..., 0:1]
        bboxes = self.cvt_bbox(label_bboxes[..., 1:], (scale_height, scale_width))
        label_bboxes = torch.concat([labels, bboxes], dim=-1)
        bboxes_empty = torch.full((64 - label_bboxes.shape[0], 5), -1)
        label_bboxes = torch.concat([label_bboxes, bboxes_empty])
        return cvt_tensor(image), cvt_tensor(label_bboxes)

    @staticmethod
    def cvt_bbox(box: Tensor, scale: tuple):
        """

        :param scale:
        :param box: [x1,x2,y1,y2]
        :return:
        """
        hs, ws = scale
        x1 = box[..., 0:1] / ws
        x2 = box[..., 1:2] / ws
        y1 = box[..., 2:3] / hs
        y2 = box[..., 3:4] / hs
        return torch.concat([y1, x1, y2, x2], dim=-1)

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
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    for images, targets in dataloader:
        print(images)
        print(targets)
        break
