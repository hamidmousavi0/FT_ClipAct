
import torch
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
def load_perturb_mnist():
    PATH = "perturb_data/perturb_mnist/"
    batch_sz = 16
    img_sz = 32  # the target size after cropping
    val_dataset = datasets.ImageFolder(PATH, transforms.Compose([
        transforms.Grayscale(),
        transforms.CenterCrop(img_sz),
        transforms.ToTensor(),
    ]))  # loading the dataset
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, shuffle=True)
    class_names = val_dataset.classes  # Getting class labels in order to use it later for accuracy.
    return val_loader, class_names
def load_imagenet(folder_not_avail=False):
    if folder_not_avail:
        val_idx = pd.read_csv('data/val/ILSVRC2012_validation_ground_truth.txt')
        print(val_idx)
        PATH = "data/val/"
        for label in list(val_idx['label'].unique()):
            temp = str(label)
            os.makedirs('{}{}'.format(PATH,temp))
        for i, label in enumerate(val_idx['label']):
            temp1 = 'ILSVRC2012_val_' + str(i + 1).zfill(8) + '.JPEG'
            temp2 = str(label)
            os.rename('{}{}'.format(PATH,temp1), '{}{}/{}'.format(PATH,temp2,temp1))
        image_name = 'ILSVRC2012_val_' + str(5300).zfill(8) + '.JPEG'
        folder_name = '197'
        img = plt.imread('{}{}/{}'.format(PATH,folder_name,image_name))
        plt.imshow(img)
        plt.show
    else:
        PATH = "data/val/"
        image_name = 'ILSVRC2012_val_' + str(5300).zfill(8) + '.JPEG'
        folder_name = '197'
        img = plt.imread(f'{PATH}{folder_name}/{image_name}')
        plt.imshow(img)
        plt.show()
        batch_sz = 2
        img_sz = 224  # the target size after cropping
        val_dataset = datasets.ImageFolder(PATH, transforms.Compose([
            transforms.CenterCrop(img_sz),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))  # loading the dataset
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, shuffle=True)
        class_names=val_dataset.classes# Getting class labels in order to use it later for accuracy.
        return val_loader,class_names
def load_curropt_mnist(name):
    BATCH_SIZE = 32
    # define transforms
    if name=="noise":
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                         transforms.ToTensor(),
                                         AddGaussianNoise(0.0,0.1)])
    elif name=="rotate":
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.RandomRotation(5)])
    elif name=="blur":
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.GaussianBlur(5)])
    elif name=="erase":
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.RandomErasing()])


    # download and create datasets
    train_dataset = datasets.MNIST(root='./data',
                                   train=True,
                                   transform=transform,
                                   download=True)

    valid_dataset = datasets.MNIST(root='./data',
                                   train=False,
                                   transform=transform)

    # define the data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)
    return train_loader, valid_loader
def load_mnist():
    BATCH_SIZE = 32
    # define transforms
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor()])

    # download and create datasets
    train_dataset = datasets.MNIST(root='./data',
                                   train=True,
                                   transform=transform,
                                   download=True)

    valid_dataset = datasets.MNIST(root='./data',
                                   train=False,
                                   transform=transform)

    # define the data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)
    return train_loader,valid_loader
def load_Cifar10():
    BATCH_SIZE = 64
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # TODO use the same image across the whole batch
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )
    return  train_loader,val_loader
if __name__ == '__main__':
    train,test = load_curropt_mnist('erase')
    i=0
    for im , la in test:
        print(im[0,:,:,:].shape)
        plt.imshow(im[0,:,:,:].squeeze())
        plt.show()
        i+=1
        if i==10:
           break
