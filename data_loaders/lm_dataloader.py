import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np


def data_loader(data_dir, transform_compose, batch_size=20, valid_size=0.2):

    def npy_loader(img_path):
        sample = torch.from_numpy(np.load(img_path))
        return sample
    train_dir = data_dir + '/train/'
    test_dir = data_dir + '/test/'

    transform_compose = transforms.Compose([
        transforms.Resize((224,224)), # anh chuan dau vao 224 x 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # train_dataset = datasets.DatasetFolder(
    #     root=train_dir,
    #     loader=npy_loader,
    #     extensions=('.npy'))
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform_compose
    )

    # test_dataset = datasets.DatasetFolder(
    #     root=test_dir,
    #     loader=npy_loader,
    #     extensions=('.npy'))
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transform_compose
    )

    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load

    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=num_workers, shuffle=False)

    return train_loader, valid_loader, test_loader, train_dataset.classes
