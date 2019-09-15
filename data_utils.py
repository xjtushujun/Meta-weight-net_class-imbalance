import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import numpy as np
import copy


def build_dataset(dataset,num_meta):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10('../data', train=False, transform=transform_test)
        img_num_list = [num_meta] * 10
        num_classes = 10

    if dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100('../data', train=False, transform=transform_test)
        img_num_list = [num_meta] * 100
        num_classes = 100

    data_list_val = {}
    for j in range(num_classes):
        data_list_val[j] = [i for i, label in enumerate(train_dataset.train_labels) if label == j]

    idx_to_meta = []
    idx_to_train = []
    print(img_num_list)

    for cls_idx, img_id_list in data_list_val.items():
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        idx_to_meta.extend(img_id_list[:img_num])
        idx_to_train.extend(img_id_list[img_num:])
    train_data = copy.deepcopy(train_dataset)
    train_data_meta = copy.deepcopy(train_dataset)
    train_data_meta.train_data = np.delete(train_dataset.train_data,idx_to_train,axis=0)
    train_data_meta.train_labels = np.delete(train_dataset.train_labels, idx_to_train, axis=0)
    train_data.train_data = np.delete(train_dataset.train_data, idx_to_meta, axis=0)
    train_data.train_labels = np.delete(train_dataset.train_labels, idx_to_meta, axis=0)

    return train_data_meta,train_data,test_dataset


def get_img_num_per_cls(dataset,imb_factor=None,num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'cifar10':
        img_max = (50000-num_meta)/10
        cls_num = 10

    if dataset == 'cifar100':
        img_max = (50000-num_meta)/100
        cls_num = 100

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls

