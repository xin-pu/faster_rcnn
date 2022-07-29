import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


class LinearDataSet(Dataset):

    def __init__(self):

        csv_data = pd.read_csv(r"..\data\CathyDemo.txt", header=None, sep='\t').iloc[:, :].values
        self.data = np.array(csv_data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, 0], self.data[index, 1]

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            if key == "image_files" or key == "annot_files":
                pass
            else:
                info += "{}:\t{}\r\n".format(key, value)
        return info


class Cathy(nn.Module):
    def __init__(self):
        super(Cathy, self).__init__()
        self.A = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        self.B = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        self.k = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = torch.nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        x = self.A * torch.sin(self.k * x + self.b) + self.B
        return x

    def get_parameters(self):
        return [self.A.data.item(), self.B.data.item(), self.k.data.item(), self.b.data.item()]


if __name__ == "__main__":
    net = Cathy().cuda()

    train_ds = LinearDataSet()
    dataloader = DataLoader(train_ds, batch_size=250, shuffle=True)

    criterion = nn.MSELoss().cuda()
    optimizer = optim.NAdam(net.parameters(), lr=0.01)

    for epoch in range(15000):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            ave_loss = running_loss / (i + 1)

            print(end="\rEpoch: {:05d}\tbatch: {:05d}\tloss: {:>.4f}\t{}".format(epoch, i, ave_loss,
                                                                                 net.get_parameters()))
            print("\r\n")
