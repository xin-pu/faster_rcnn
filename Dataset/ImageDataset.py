import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import *
import pandas as pd
import cv2


class ImageDataSet(Dataset):

    def __init__(self,
                 img_index_file,
                 annotation_folder,
                 input_size=(416, 416),
                 enhance=False):
        """
        构造目标检测的数据集
        :param img_index_file: 含有数据集 图像路径的索引文件
        :param annotation_folder: 含有以图像名命名的检测框信息的目录
        :param input_size: 图像输入尺寸
        :param enhance: 是否数据增强
        """
        self.img_index_file = img_index_file
        self.annotation_folder = annotation_folder
        self.imageSize = input_size
        self.enhance = enhance
        self.image_filenames = pd.read_csv(img_index_file, header=None)
        self.len = self.image_filenames.__len__()

        self.transform = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_file = self.image_filenames.iloc[index, 0]
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image = self.image_transform(image)
        return image, torch.zeros(3)

    def image_transform(self, image):
        """

        :rtype: object
        """
        image = cv2.resize(image, self.imageSize)

        if self.enhance:
            image = self.transform(image)
        image = ToTensor()(image)
        return image

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "{}:\t{}\r\n".format(key, value)
        return info


if __name__ == "__main__":
    img_indexed_file = r"E:\Code DeepLearning\darknet_release\myData\myData_test.txt"
    annotations_folder = r"E:\Code DeepLearning\darknet_release\myData\labels"
    dataset = ImageDataSet(img_indexed_file, annotations_folder)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(dataset[1])

    for images, targets in dataloader:
        print(images)
        print(targets)
        print("-" * 20)
