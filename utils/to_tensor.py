import torch
from torch import Tensor
from torch.nn import Module

cpu = False


def cvt_tensor(tensor: Tensor):
    if cpu:
        return tensor.cpu()
    return tensor.cuda() if (torch.cuda.is_available()) else tensor.cpu()


def cvt_module(module: Module):
    if cpu:
        return module.cpu()
    return module.cuda() if (torch.cuda.is_available()) else module.cpu()


if __name__ == "__main__":
    t = torch.tensor([1, 2]).cuda()
    print(cvt_tensor(t))

    d = torch.tensor((2, 3))
    print(cvt_tensor(d))

    a = torch.asarray([[2, 0, 3, 8]]).float()
    b = torch.asarray([3]).long()
    loss = torch.nn.functional.cross_entropy(a, b)
    print(loss)
