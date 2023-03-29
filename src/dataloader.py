import os
import numpy as np

import torch
from torch.utils.data import Dataset

from utils.utils import load_nii

np.random.seed(0)


class DataGenerator(Dataset):
    def __init__(self, config, list_IDs):
        self.classes = os.listdir(config.data.data_path)
        self.list_IDs = list_IDs
        self.n_channels = config.data.number_of_channels
        self.shuffle = config.data.shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()
        self.data_path = config.data.data_path
        self.deep_unet = config.model.depth_unet
        self.number_classes = config.data.number_classes

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        X, y = self.__data_generation(self.list_IDs[self.indexes[index]])
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """
        path = list_IDs_temp[0]
        path_gt = list_IDs_temp[1]
        image_size = np.shape(load_nii(path)[0])

        X1 = resize_for_unet(load_nii(path)[0], image_size, self.deep_unet)
        y1 = resize_for_unet(load_nii(path_gt)[0], image_size, self.deep_unet)

        image_size = (image_size[0] - image_size[0] % (2 ** self.deep_unet),
                      image_size[1] - image_size[1] % (2 ** self.deep_unet),
                      image_size[2])

        X = np.empty((self.n_channels, *image_size))
        y = np.empty(image_size, dtype=np.int64)

        X[0, :, :, :] = X1
        y[:, :, :] = y1

        X = torch.tensor(X, dtype=torch.float)
        y = torch.nn.functional.one_hot(torch.tensor(y), num_classes=self.number_classes)
        y = y.type(torch.float)
        return X, y


def get_data(data_path):
    """
    recovers all MRI paths
    """
    image_paths = []
    for patient in os.listdir(data_path):
        if patient[-1] != 'x' and patient[-1] != 'y':
            # There are 2 frames and 2 labels in each folder
            frame1 = [0, 1]
            frame2 = [0, 1]
            for frame in os.listdir(os.path.join(data_path, patient)):
                if len(frame) > 15:
                    if len(frame) == 25 and frame[17] == '1':
                        frame1[0] = os.path.join(os.path.join(data_path, patient), frame)
                    elif len(frame) == 28 and frame[17] == '1':
                        frame1[1] = os.path.join(os.path.join(data_path, patient), frame)
                    elif len(frame) == 25 and frame[17] == '2':
                        frame2[0] = os.path.join(os.path.join(data_path, patient), frame)
                    elif len(frame) == 28 and frame[17] == '2':
                        frame2[1] = os.path.join(os.path.join(data_path, patient), frame)
            image_paths.append(frame1 + frame2)
    return image_paths


def data_split(config, paths_list):
    """
    Splits the paths list into three for train, val and test
    """
    n = len(paths_list)
    split_1 = int(n * config.data.train_split)
    split_2 = split_1 + int(n * config.data.val_split)
    return paths_list[:split_1], paths_list[split_1:split_2], paths_list[split_2:]


def reshape(L):
    """
    Transforms a matrix of size (n,4) into a matrix of size (2n,2)
    """
    l = []
    for i in range(len(L)):
        l.append(L[i][:2])
        l.append(L[i][2:])
    l = np.array(l)
    return l


def resize_for_unet(X, image_size, deep_unet):
    """
    Takes 2 tensors of order 3 with the same shape: image_size and trims the edges so that the size (height & width)
    of each image is a multiple of 2^deep_unet
    """

    resize_x = image_size[0] % (2 ** deep_unet)
    resize_y = image_size[1] % (2 ** deep_unet)

    x_right = resize_x // 2
    x_left = resize_x - x_right
    y_top = resize_y // 2
    y_bottom = resize_y - y_top

    if (resize_x, resize_y) == (0, 0):
        return X

    return X[x_right:image_size[0] - x_left, y_top:image_size[1] - y_bottom, :]


def create_generators(config):
    """Returns three generators"""
    image_paths = get_data(config.data.data_path)  # Get a list of data's path

    train_list, val_list, test_list = data_split(config, np.asarray(image_paths))  # split frame into 3 parts
    train_list = reshape(train_list)
    val_list = reshape(val_list)
    test_list = reshape(test_list)

    train_generator = DataGenerator(config, train_list)
    validation_generator = DataGenerator(config, val_list)
    test_generator = DataGenerator(config, test_list)

    return train_generator, validation_generator, test_generator