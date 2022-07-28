import numpy as np
import torch as torch





def to_tensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()

    if cuda:

        tensor_cuda = tensor.cuda()
    return tensor
