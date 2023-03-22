import utils
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from deepreduce_models.resnet import *
from minionn_torch.minionn_model import Minionn

import torch

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
parser.add_argument('--model', type=str, default='resnet50', help='type of base model to use')
args = parser.parse_args()

num = args.model

plt.figure()
# res = torch.load('/Users/chen4384/simple-blackbox-attack/save_cifar/pixel_6approx_1000_0_32_0.2000_rand.pth')
for num in ['918K', '459K', '393K', '229K', '197K', '115K', '98K']:
    res = torch.load('/scratch/gilbreth/chen4384/RobustPNet/save/dct_{}_1000_0_14_0.2000_rand.pth'.format(num))

    print(res.keys())
    print(res['adv'].shape)
    print(res['succs'][:, -1].sum())
    print(res['queries'].sum(dim=1).mean())
    print(res['probs'].shape)
    print(res['l2_norms'].shape)
    print(res['linf_norms'].shape)

    y = [res['succs'][:, i].sum() / 1000 for i in range(res['succs'].shape[1])]
    x = []
    ssum = 0
    for i in range(res['queries'].shape[1]):
        ssum += res['queries'][:, i].sum()
        x.append(ssum / 1000)

    plt.plot(x, y, label=num)
plt.title('SimBA-DCT for DeepReDuce Models on TinyImageNet')
plt.grid()
plt.ylabel('Attack Success Rate')
plt.xlabel('Queries')
plt.legend()
plt.savefig('./asr_vs_queries.png')

checkpoint = torch.load('save/images_{}_1000.pth'.format(num))
images = checkpoint['images'][:50]
labels = checkpoint['labels'][:50]
#
# plt.imsave('adv.png', res['adv'][2].numpy().transpose(1, 2, 0))
# plt.imsave('ori.png', images[2].numpy().transpose(1, 2, 0))
#
# model = Minionn()
# model.load_state_dict(torch.load('minionn_torch/pretrained/7_approx.pt'))

model = globals().get('DRD_TINY_' + num)(num_classes=200).cuda()
checkpoint = torch.load('deepreduce_models/TinyImageNet_models/model_DRD_TINY_{}.pth.tar'.format(num))

keys = checkpoint["snet"].keys()
values = checkpoint["snet"].values()

new_keys = []
for key in keys:
    new_key = key[7:]
    new_keys.append(new_key)
new_dict = OrderedDict(list(zip(new_keys, values)))
model.load_state_dict(new_dict)
model.eval()

print(labels)
outputs_1 = model(utils.apply_normalization(images.cuda(), 'tinyimagenet'))
outputs_2 = model(utils.apply_normalization(res['adv'][:50].cuda(), 'tinyimagenet'))
_, predicted_1 = torch.max(outputs_1.data, 1)
_, predicted_2 = torch.max(outputs_2.data, 1)
print(predicted_1)
print(predicted_2)
print(sum(predicted_1 == predicted_2))
