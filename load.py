import utils
import argparse
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from collections import OrderedDict

from deepreduce_models.resnet import *
from minionn_torch.minionn_model import Minionn

import torch

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
# parser.add_argument('--model', type=str, default='resnet50', help='type of base model to use')
parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
args = parser.parse_args()

# num = args.model

plt.figure()
# res = torch.load('/Users/chen4384/simple-blackbox-attack/save_cifar/pixel_6approx_1000_0_32_0.2000_rand.pth')
# nums = ['918K', '459K', '393K', '229K', '197K', '115K', '98K']
nums = ['relu', '3_approx', '5_approx', '6_approx', '7_approx']

p1, p2 = '', 'dct'
if args.targeted:
    p1 = 'targeted_'
if args.pixel_attack:
    p2 = 'pixel'

p3, p4 = 'Untargeted', 'DCT'
if args.targeted:
    p3 = 'Targeted'
if args.pixel_attack:
    p4 = 'Pixel'

p5 = '14'
if args.pixel_attack:
    p5 = '64'

for num in nums:
    print('# of ReLUs:', num)
    # res = torch.load('/scratch/gilbreth/chen4384/RobustPNet/save/{}_{}{}_1000_0_{}_0.2000_rand.pth'.format(p2, p1, num, p5))
    print('# :', num)
    res = torch.load('/scratch/gilbreth/chen4384/RobustPNet/save_cifar/{}_{}{}_1000_0_32_0.2000_rand.pth'.format(p2, p1, num))

    print(res.keys())
    print(res['adv'].shape)
    print(res['succs'][:, -1].sum())
    print(res['queries'].sum(dim=1).mean())
    print(res['probs'].shape)
    print(res['l2_norms'].shape)
    print(res['linf_norms'].shape)
    # print(res['succs'][:, -1] == res['succs'][:, -2])

    y = [res['succs'][:, i].mean() for i in range(res['succs'].shape[1])]

    x = []
    ssum = 0
    for i in range(res['queries'].shape[1]):
        ssum += res['queries'][:, i].sum()
        x.append(ssum / 1000)

    # bspline = interpolate.make_interp_spline(x, y)
    # x_new = np.linspace(min(x), max(x), 500)
    # y_new = bspline(x_new)
    # print(x[-50:])
    assert all(x[i] <= x[i + 1] for i in range(len(x) - 1))
    # print(y[-20:])
    assert all(y[i] <= y[i + 1] for i in range(len(y) - 1))
    plt.plot(x, y, label=num)

# plt.title('{} SimBA-{} for DRD Models on TinyImageNet'.format(p3, p4))
plt.title('{} SimBA-{} for MiniONN Models on CIFAR10'.format(p3, p4))
plt.grid()
plt.ylabel('Attack Success Rate')
plt.xlabel('Queries')
plt.legend()
plt.savefig('./{}_{}_asr_vs_queries_minionn.png'.format(p3, p4))

# checkpoint = torch.load('save/images_{}_1000.pth'.format(num))
# images = checkpoint['images'][:50]
# labels = checkpoint['labels'][:50]
#
# plt.imsave('adv.png', res['adv'][2].numpy().transpose(1, 2, 0))
# plt.imsave('ori.png', images[2].numpy().transpose(1, 2, 0))
#
# model = Minionn()
# model.load_state_dict(torch.load('minionn_torch/pretrained/7_approx.pt'))

# model = globals().get('DRD_TINY_' + num)(num_classes=200).cuda()
# checkpoint = torch.load('deepreduce_models/TinyImageNet_models/model_DRD_TINY_{}.pth.tar'.format(num))
#
# keys = checkpoint["snet"].keys()
# values = checkpoint["snet"].values()
#
# new_keys = []
# for key in keys:
#     new_key = key[7:]
#     new_keys.append(new_key)
# new_dict = OrderedDict(list(zip(new_keys, values)))
# model.load_state_dict(new_dict)
# model.eval()
#
# print(labels)
# outputs_1 = model(utils.apply_normalization(images.cuda(), 'tinyimagenet'))
# outputs_2 = model(utils.apply_normalization(res['adv'][:50].cuda(), 'tinyimagenet'))
# _, predicted_1 = torch.max(outputs_1.data, 1)
# _, predicted_2 = torch.max(outputs_2.data, 1)
# print(predicted_1)
# print(predicted_2)
# print(sum(predicted_1 == predicted_2))
