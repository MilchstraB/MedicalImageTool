import os
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MedicalDataset(Dataset):
    """
    Pass in a custom dataset that conforms to the format.
    Args:
        transforms (list): The compose of transforms for image.
        dataset_root (str): The dataset directory.
        num_classes (int): Number of classes.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        Examples:
            import medicalseg.transforms as T
            from paddleseg.datasets import MedicalDataset
            transforms = [T.RandomRotation3D(degrees=90)]
            dataset_root = 'dataset_root_path'
            dataset = MedicalDataset(transforms = transforms,
                              dataset_root = dataset_root,
                              num_classes = 3,
                              mode = 'train')
            for data in dataset:
                img, label = data
                print(img.shape, label.shape)
                print(np.unique(label))
    """

    def __init__(self,
                 dataset_root,
                 transforms,
                 num_classes,
                 mode='train'):
        self.dataset_root = dataset_root
        self.transforms = transforms
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes

        # Using text file to load file path.The path should be full path.
        if mode == 'train':
            file_path = os.path.join(self.dataset_root, 'train_list.txt')
        elif mode == 'val':
            file_path = os.path.join(self.dataset_root, 'val_list.txt')
        elif mode == 'test':
            file_path = os.path.join(self.dataset_root, 'test_list.txt')
        else:
            raise ValueError(
                "`mode` should be 'train', 'val' or 'test', but got {}.".
                format(mode))

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 2:
                    raise Exception("File list format incorrect! It should be"
                                    " image_name label_name\\n")
                else:
                    image_path = items[0]
                    grt_path = items[1]
                self.file_list.append([image_path, grt_path])

        # come from shiyutang: 没有看懂
        if mode == 'train':
            self.file_list = self.file_list * 10

    def __getitem__(self, index):
        image_path, label_path = self.file_list[index]
        # Assuming that the data is memorized as npy
        image = np.load(image_path)
        label = np.load(label_path)
        label = label['arr_0']

        image = self.transforms(image)
        label = self.transforms(label)

        image = torch.tensor(image, dtype=torch.float32)
        image = torch.unsqueeze(image, dim=0)
        label = torch.tensor(label)

        return image, label

    def __len__(self):
        return len(self.file_list)
