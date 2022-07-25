import yaml


class TrainPlan(object):
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        cfg = self.get_dataset_cfg(cfg_file)
        self.image_index_file = cfg["image_index_file"]
        self.annot_encode_folder = cfg["annot_encode_folder"]
        self.labels = cfg["labels"]
        self.min_size = cfg["min_size"]
        self.max_size = cfg["max_size"]

        self.enhance = cfg["enhance"]

        self.epoch = cfg["epoch"]
        self.batch_size = cfg["batch_size"]
        self.val_split = cfg["val_split"]

        self.rpn_sigma = cfg["rpn_sigma"]
        self.roi_sigma = cfg["roi_sigma"]

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "{}:\t{}\r\n".format(key, value)
        return info

    @staticmethod
    def get_dataset_cfg(cfg_file):
        with open(cfg_file, 'r') as file:
            cfg = yaml.safe_load(file)
            return cfg


if __name__ == "__main__":
    cfg_obj = TrainPlan("voc_train.yml")
    print(cfg_obj)
