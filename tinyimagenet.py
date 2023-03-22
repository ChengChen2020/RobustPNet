import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torchvision
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

"""
TinyImageNet
64 * 64
Train 200 classes * 500 samples/class = 100,000
Test  10,000
Val   10,000
"""


def tin_val_loader():
    data_dir = './data/tiny-imagenet-200/'
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
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=256, shuffle=True, num_workers=num_workers[x])
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    #       label_id  --> label
    # e.g., n01443537 --> goldfish, Carassius auratus
    small_labels = {}
    with open(os.path.join(data_dir, "words.txt"), "r") as dictionary_file:
        line = dictionary_file.readline()
        while line:
            label_id, label = line.strip().split("\t")
            small_labels[label_id] = label
            line = dictionary_file.readline()

    # print(list(small_labels.items())[:5])

    train_loader = dataloaders['train']

    # print(len(train_loader))
    # print(train_loader.dataset.class_to_idx['n12267677'])

    labels = {}
    label_ids = {}
    for label_index, label_id in enumerate(train_loader.dataset.classes):
        label = small_labels[label_id]
        labels[label_index] = label
        label_ids[label_id] = label_index

    # print(list(labels.items())[:5])
    # print(list(label_ids.items())[:5])

    val_label_map = {}
    with open(os.path.join(data_dir, "val/val_annotations.txt"), "r") as val_label_file:
        line = val_label_file.readline()
        while line:
            file_name, label_id, _, _, _, _ = line.strip().split("\t")
            val_label_map[file_name] = label_id
            line = val_label_file.readline()

    """
    `imgs`: This attribute is a list of tuples, 
    where each tuple contains the image path and its corresponding label.
    """
    val_loader = dataloaders['val']
    for i in range(len(val_loader.dataset.imgs)):
        file_path = val_loader.dataset.imgs[i][0]

        file_name = os.path.basename(file_path)
        label_id = val_label_map[file_name]

        val_loader.dataset.imgs[i] = (file_path, label_ids[label_id])

    return image_datasets['val'], val_loader


if __name__ == '__main__':

    _, val_loader = tin_val_loader()

    for images, labels in val_loader:
        print(images.shape, labels.shape)
        break
