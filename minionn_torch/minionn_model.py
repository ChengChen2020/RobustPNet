import h5py
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms


class Poly(nn.Module):
    def __init__(self):
        super(Poly, self).__init__()

    def forward(self, x):
        return .1992 + .5002*x + .1997*x**2


ACTIVATION_DICT = {
    0: [],
    1: [6],
    2: [5, 6],
    3: [0, 2, 3],
    5: [0, 1, 3, 5, 6],
    6: [0, 1, 2, 3, 5, 6],
    7: [0, 1, 2, 3, 4, 5, 6]
}


class Minionn(nn.Module):
    def __init__(self, approx=0):
        super(Minionn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.conv7 = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.pool = nn.AvgPool2d(2, stride=2)
        self.flat = nn.Flatten()
        self.act = [nn.ReLU6() if i not in ACTIVATION_DICT[approx] else Poly() for i in range(7)]
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = self.act[0](self.conv1(x))
        x = self.pool(self.act[1](self.conv2(x)))
        x = self.act[2](self.conv3(x))
        x = self.pool(self.act[3](self.conv4(x)))
        x = self.act[4](self.conv5(x))
        x = self.act[5](self.conv6(x))
        x = self.act[6](self.conv7(x))
        x = self.flat(x.permute(0, 2, 3, 1))
        x = self.fc(x)
        return nn.functional.softmax(x, dim=1)


def get_num_params(net):
    """Return the number of parameters of the net"""
    assert(isinstance(net, torch.nn.Module))
    num = 0
    for p in list(net.parameters()):
        n = 1
        for s in list(p.size()):
            n *= s
        num += n
    return num


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for approx in [0, 3, 5, 6, 7]:
        model = Minionn(approx=approx).to(device)
        if approx == 0:
            model.load_state_dict(torch.load('pretrained/relu.pt'))
        else:
            model.load_state_dict(torch.load('pretrained/' + str(approx) + '_approx.pt'))
        model.eval()

        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                               download=True, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                 shuffle=False, num_workers=1)

        correct = 0
        total = 0
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(approx, 100 * correct / total)

    # torch.save(model.state_dict(), '6_approx.pt')


# 0: 85.02
# 3: 85.87
# 5: 85.33
# 6: 83.06
# 7: 82.95

    # model = Minionn(approx=7)
    #
    # a = h5py.File('/Users/chen4384/simple-blackbox-attack/minionn_pretrained/model7.h5', 'r')
    #
    # print(a.keys())
    #
    # model.conv1.weight.data = torch.from_numpy(a['conv2d']['conv2d']['kernel:0'][()]).permute(3, 2, 0, 1)
    # model.conv2.weight.data = torch.from_numpy(a['conv2d_1']['conv2d_1']['kernel:0'][()]).permute(3, 2, 0, 1)
    # model.conv3.weight.data = torch.from_numpy(a['conv2d_2']['conv2d_2']['kernel:0'][()]).permute(3, 2, 0, 1)
    # model.conv4.weight.data = torch.from_numpy(a['conv2d_3']['conv2d_3']['kernel:0'][()]).permute(3, 2, 0, 1)
    # model.conv5.weight.data = torch.from_numpy(a['conv2d_4']['conv2d_4']['kernel:0'][()]).permute(3, 2, 0, 1)
    # model.conv6.weight.data = torch.from_numpy(a['conv2d_5']['conv2d_5']['kernel:0'][()]).permute(3, 2, 0, 1)
    # model.conv7.weight.data = torch.from_numpy(a['conv2d_6']['conv2d_6']['kernel:0'][()]).permute(3, 2, 0, 1)
    # model.conv1.bias.data = torch.from_numpy(a['conv2d']['conv2d']['bias:0'][()])
    # model.conv2.bias.data = torch.from_numpy(a['conv2d_1']['conv2d_1']['bias:0'][()])
    # model.conv3.bias.data = torch.from_numpy(a['conv2d_2']['conv2d_2']['bias:0'][()])
    # model.conv4.bias.data = torch.from_numpy(a['conv2d_3']['conv2d_3']['bias:0'][()])
    # model.conv5.bias.data = torch.from_numpy(a['conv2d_4']['conv2d_4']['bias:0'][()])
    # model.conv6.bias.data = torch.from_numpy(a['conv2d_5']['conv2d_5']['bias:0'][()])
    # model.conv7.bias.data = torch.from_numpy(a['conv2d_6']['conv2d_6']['bias:0'][()])
    #
    # model.fc.weight.data = torch.from_numpy(a['dense']['dense']['kernel:0'][()]).permute(1, 0)
    # model.fc.bias.data = torch.from_numpy(a['dense']['dense']['bias:0'][()])
    #
    # print(get_num_params(model))
