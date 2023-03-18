import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torchvision
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from resnet import *

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    # (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    testset = datasets.CIFAR100(root='../data', train=False,
                                download=True, transform=transform)

    data_dir = 'tiny-imagenet-200/'
    num_workers = {'train': 0, 'val': 0, 'test': 0}
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=512, shuffle=True, num_workers=num_workers[x])
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    small_labels = {}
    with open(os.path.join(data_dir, "words.txt"), "r") as dictionary_file:
        line = dictionary_file.readline()
        while line:
            label_id, label = line.strip().split("\t")
            small_labels[label_id] = label
            line = dictionary_file.readline()

    print(list(small_labels.items())[:5])

    train_loader = dataloaders['train']

    labels = {}
    label_ids = {}
    for label_index, label_id in enumerate(train_loader.dataset.classes):
        label = small_labels[label_id]
        labels[label_index] = label
        label_ids[label_id] = label_index

    print(list(labels.items())[:5])
    print(list(label_ids.items())[:5])

    val_label_map = {}
    with open(os.path.join(data_dir, "val/val_annotations.txt"), "r") as val_label_file:
        line = val_label_file.readline()
        while line:
            file_name, label_id, _, _, _, _ = line.strip().split("\t")
            val_label_map[file_name] = label_id
            line = val_label_file.readline()

    val_loader = dataloaders['val']
    for i in range(len(val_loader.dataset.imgs)):
        file_path = val_loader.dataset.imgs[i][0]

        file_name = os.path.basename(file_path)
        label_id = val_label_map[file_name]

        val_loader.dataset.imgs[i] = (file_path, label_ids[label_id])

    # """ CIFAR-100 models """
    # for num in ['230K', '115K', '57K', '49K', '29K', '14K', '12K', '7K']:
    #     model = globals().get('DRD_C100_' + num)(num_classes=100).to(device)
    #     checkpoint = torch.load('CIFAR100_models/model_DRD_C100_{}.pth.tar'.format(num))
    #
    #     testloader = data.DataLoader(testset, batch_size=512,
    #                                  shuffle=False, num_workers=0)

    """ TinyImageNet models """
    for num in ['918K', '459K', '393K', '229K', '197K', '115K', '98K']:
        model = globals().get('DRD_TINY_' + num)(num_classes=200).to(device)
        checkpoint = torch.load('TinyImageNet_models/model_DRD_TINY_{}.pth.tar'.format(num))

        testloader = dataloaders['val']

        keys = checkpoint["snet"].keys()
        values = checkpoint["snet"].values()

        new_keys = []
        for key in keys:
            new_key = key[7:]
            new_keys.append(new_key)
        new_dict = OrderedDict(list(zip(new_keys, values)))
        model.load_state_dict(new_dict)

        total = 0
        correct = 0
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(100 * correct / total)
