import torch
from torch import Tensor


def to_device(tensor: Tensor):
    return tensor.cuda() if (torch.cuda.is_available()) else tensor.cpu()


if __name__ == "__main__":
    t = torch.tensor([1, 2]).cuda()
    print(to_device(t))

    d = torch.tensor((2, 3))
    print(to_device(d))
