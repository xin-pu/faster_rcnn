import os

import pandas as pd
import tensorflow as tf

from cfg.plan_config import TrainPlan


def read_features(path, min_size, max_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    w, h, c = img.shape
    scale1 = min_size / min(h, w)
    scale2 = max_size / max(h, w)
    scale = min(scale1, scale2)
    img = tf.image.resize(img, (w * scale, h * scale))
    img = img / 255.
    return img


class DatasetGenerator(object):
    def __init__(self, train_plan):
        self.train_plan = train_plan
        image_files = pd.read_csv(train_plan.image_index_file, header=None).iloc[:, 0]
        annot_files = self.get_annot_file(image_files)
        ds_image_path = tf.data.Dataset.from_tensor_slices((image_files, annot_files))
        self.ds_image = ds_image_path.map(self.wrap_read_dataset)

    def get_annot_file(self, image_files):
        ann = []
        for f in image_files:
            file_name, extension = os.path.splitext(os.path.basename(f))
            ann.append(os.path.join(self.train_plan.annot_encode_folder, "{}.txt".format(file_name)))
        return ann

    def wrap_read_dataset(self, img_file, ann_file):
        min_size = self.train_plan.min_size
        max_size = self.train_plan.max_size
        img = tf.py_function(read_features, inp=[img_file, min_size, max_size], Tout=[tf.float32])
        return img[0]

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "{}:\t{}\r\n".format(key, value)
        return info


if __name__ == "__main__":
    tp = TrainPlan("../cfg/voc_train.yml")
    data_g = DatasetGenerator(tp)
    print(data_g)
    for x in data_g.ds_image.take(1):
        print(x)
