import tensorflow as tf


class DatasetGenerator(object):
    def __init__(self, train_plan):
        self.train_plan = train_plan

    def read_features(self, path):
        return tf.ones(shape=[416, 416, 3])

    def encode_labels(self, path):
        return tf.ones(shape=[20])

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "{}:\t{}\r\n".format(key, value)
        return info


if __name__ == "__main__":
    data_g = DatasetGenerator("", "")
    print(data_g)
