import torch
from torch import Tensor
from torch.nn import Module

cpu = False


def to_device(tensor: Tensor):
    if cpu:
        return tensor.cpu()
    return tensor.cuda() if (torch.cuda.is_available()) else tensor.cpu()


def cvt_module(module: Module):
    if cpu:
        return module.cpu()
    return module.cuda() if (torch.cuda.is_available()) else module.cpu()


if __name__ == "__main__":
    t = torch.tensor([1, 2]).cuda()
    print(to_device(t))

    d = torch.tensor((2, 3))
    print(to_device(d))
