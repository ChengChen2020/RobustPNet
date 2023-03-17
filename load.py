import utils
import torch
import matplotlib.pyplot as plt

from minionn_torch.minionn_model import Minionn

res = torch.load('/Users/chen4384/simple-blackbox-attack/save_cifar/pixel_6approx_1000_0_32_0.2000_rand.pth')

print(res.keys())
print(res['adv'].shape)
print(res['succs'][:, -1].sum())
print(res['queries'].sum(dim=1).mean())
print(res['probs'][2])
print(res['l2_norms'].shape)
print(res['linf_norms'].shape)

# checkpoint = torch.load('save_cifar/images_7approx_1000.pth')
# images = checkpoint['images']
# labels = checkpoint['labels']
#
# plt.imsave('adv.png', res['adv'][2].numpy().transpose(1, 2, 0))
# plt.imsave('ori.png', images[2].numpy().transpose(1, 2, 0))
#
# model = Minionn()
# model.load_state_dict(torch.load('minionn_torch/pretrained/7_approx.pt'))
#
# print(model(utils.apply_normalization(res['adv'], 'cifar'))[2])
# print(model(utils.apply_normalization(images, 'cifar'))[2])
