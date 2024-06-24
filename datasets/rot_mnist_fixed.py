# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.MNISTMLP import MNISTMLP

from datasets.perm_mnist import store_mnist_loaders
from datasets.transforms.rotation import GivenRotation
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace


class RotatedMNISTFixed(ContinualDataset):
    NAME = 'rot-mnist-fixed'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 20

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test.sh lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        super(RotatedMNISTFixed, self).__init__(args)
        RotatedMNISTFixed.N_TASKS = args.n_tasks_mnist
        lst_degrees = [args.deg_inc * i for i in range(args.n_tasks_mnist)]
        self.rotations = [GivenRotation(deg) for deg in lst_degrees]
        self.task_id = 0

    def get_data_loaders(self):
        transform = transforms.Compose((self.rotations[self.task_id], transforms.ToTensor()))
        train, test = store_mnist_loaders(transform, self)
        self.task_id += 1
        return train, test

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, RotatedMNISTFixed.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None

