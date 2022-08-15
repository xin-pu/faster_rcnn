import io
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ["raccoon"]  # 改成自己的类别


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


ann_folder = r"F:\Raccoon\annotations\all"
img_folder = r"F:\Raccoon\images"
lab_folder = r"F:\Raccoon\labels"


def convert_annotation(image_id):
    in_file = open(ann_folder + r'\{}.xml'.format(image_id))
    out_file = open(lab_folder + r'\{}.txt'.format(image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in b]) + '\n')


import os

path = img_folder  # 文件夹目录
files = os.listdir(path)
for f in files:
    print(img_folder + "\\" + f)
    # image_id = f.replace(".jpg", "")
    #
    # convert_annotation(image_id)
