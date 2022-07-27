import torch
import torchvision
from torch import nn, optim
from torchvision.transforms import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    transform = transforms.Compose(
        [ToTensor(),
         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_ds = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

    test_ds = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    net = Net().cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.NAdam(net.parameters(), lr=0.001)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
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
            print(end="\rEpoch: {:05d}\tbatch: {:05d}\tloss: {:>.4f}".format(epoch, i, ave_loss))
        print("\r\n")

    torch.save(net.state_dict(), "../save/cifar.pt")
    print('Finished Training')
