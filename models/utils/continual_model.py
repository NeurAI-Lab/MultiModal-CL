# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from argparse import Namespace
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision
from torch.optim import SGD

from utils.conf import get_device
from utils.magic import persistent_locals
from datasets.utils.continual_dataset import ContinualDataset
from utils.loggers import *

import os
import numpy as np
from typing import Tuple

try:
    import wandb
except ImportError:
    wandb = None


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.device = get_device()
        self.task = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        if wandb is not None and not self.args.nowand:
            pl = persistent_locals(self.observe)
            ret = pl(*args, **kwargs)
            self.autolog_wandb(pl.locals)
        else:
            ret = self.observe(*args, **kwargs)
        return ret

    @abstractmethod
    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

    def autolog_wandb(self, locals):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            wandb.log({k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
                      for k, v in locals.items() if k.startswith('_wandb_') or k.startswith('loss')})

    # Custom functions for Dual Memories
    def init_loggers(self, dataset):
        """
        Initialize loggers for all the models in the model.addit_models attribute
        :param dataset: the continual dataset at hand
        """
        self.loggers = {}
        self.results = {}
        self.results_mask_classes = {}

        for model_idt in self.addit_models:
            self.results[model_idt], self.results_mask_classes[model_idt] = [], []
            self.loggers[model_idt] = Logger(
                dataset.SETTING,
                dataset.NAME,
                self.NAME,
                self.args.output_dir,
                self.args.experiment_id,
                '_' + model_idt
            )

    def save_models(self, dataset):
        """
        Save the models and optimizer state dictionaries
        :param dataset: the continual dataset at hand
        """
        model_dir = os.path.join(self.args.output_dir, "results", dataset.SETTING, dataset.NAME, self.NAME, self.args.experiment_id)
        os.makedirs(model_dir, exist_ok=True)
        model_dict = {}
        model_dict['task'] = self.task
        model_dict['net'] = self.net.state_dict()
        model_dict['optimizer'] = self.opt.state_dict()

        for model_idt in self.addit_models:
            model_dict[model_idt] = getattr(self, model_idt).state_dict()

        torch.save(model_dict, os.path.join(model_dir, f'model_task{self.task}.ph'))

    def evaluate(self, dataset: ContinualDataset, last=False, eval_model='net') -> Tuple[list, list]:
        """
        Evaluates the accuracy of the model for each past task.
        :param model: the model to be evaluated
        :param dataset: the continual dataset at hand
        :param eval_model: name of the model to evaluate
        :return: a tuple of lists, containing the class-il
                 and task-il accuracy for each task
        """
        eval_model = getattr(self, eval_model)
        status = eval_model.training
        eval_model.eval()
        accs, accs_mask_classes = [], []
        for k, test_loader in enumerate(dataset.test_loaders):
            if last and k < len(dataset.test_loaders) - 1:
                continue
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            for data in test_loader:
                with torch.no_grad():
                    (audio, video), labels, _ = data
                    audio, video, labels = audio.to(self.device), video.to(self.device), labels.to(self.device)
                    if 'class-il' not in self.COMPATIBILITY:
                        outputs = eval_model(audio.unsqueeze(1), video, k)
                    else:
                        outputs = eval_model(audio.unsqueeze(1), video)

                    _, pred = torch.max(outputs.data, 1)
                    correct += torch.sum(pred == labels).item()
                    total += labels.shape[0]

                    if dataset.SETTING == 'class-il':
                        mask_classes(outputs, dataset, k)
                        _, pred = torch.max(outputs.data, 1)
                        correct_mask_classes += torch.sum(pred == labels).item()

            print(f'Task {k} Accuracy: {correct / total * 100}')
            accs.append(correct / total * 100
                        if 'class-il' in self.COMPATIBILITY else 0)
            accs_mask_classes.append(correct_mask_classes / total * 100)

        eval_model.train(status)
        return accs, accs_mask_classes

    def evaluate_gcl(self, dataset, eval_model) -> float:
        """
        Evaluates the final accuracy of the model.
        :param model: the model to be evaluated
        :param dataset: the GCL dataset at hand
        :return: a float value that indicates the accuracy
        """
        eval_model = getattr(self, eval_model)
        status = eval_model.training
        eval_model.eval()
        correct, total = 0, 0
        while not dataset.test_over:
            inputs, labels = dataset.get_test_data()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = eval_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += torch.sum(predicted == labels).item()
            total += labels.shape[0]

        acc = correct / total * 100
        eval_model.train(status)
        return acc

    def eval_before_training(self, dataset: ContinualDataset):
        if self.task and not self.args.ignore_other_metrics:
            for model_idt in self.addit_models:
                print(f'Evaluating {model_idt}')
                accs = self.evaluate(dataset, last=True, eval_model=model_idt)
                self.results[model_idt][self.task - 1] = self.results[model_idt][self.task - 1] + accs[0]

                if dataset.SETTING == 'class-il':
                    self.results_mask_classes[model_idt][self.task - 1] = self.results_mask_classes[model_idt][self.task - 1] + accs[1]

    def eval_addit_models(self, dataset: ContinualDataset):

        if dataset.SETTING == "general-continual":
            for model_idt in self.addit_models:
                print(f'Evaluating {model_idt}')
            acc = self.evaluate_gcl(deepcopy(dataset), eval_model=model_idt)
            print('Accuracy:', acc)
            if not self.args.disable_log:
                self.loggers[model_idt].log(acc)
                self.loggers[model_idt].write(vars(self.args))
            return
        else:
            for model_idt in self.addit_models:
                print(f'Evaluating {model_idt}')
                accs = self.evaluate(dataset, eval_model=model_idt)
                self.results[model_idt].append(accs[0])
                self.results_mask_classes[model_idt].append(accs[1])

                mean_acc = np.mean(accs, axis=1)
                print_mean_accuracy(mean_acc, self.task, dataset.SETTING)

                if not self.args.disable_log:
                    self.loggers[model_idt].log(mean_acc)
                    self.loggers[model_idt].log_fullacc(accs)
            # =====================================
            # Operations at end of training
            # =====================================

            if not (self.task == dataset.N_TASKS):
                return

            if not self.args.disable_log and not self.args.ignore_other_metrics:
                for model_idt in self.addit_models:
                    logger = self.loggers[model_idt]
                    logger.add_bwt(self.results[model_idt], self.results_mask_classes[model_idt])
                    logger.add_forgetting(self.results[model_idt], self.results_mask_classes[model_idt])
                    logger.add_fwt(self.results[model_idt], self.random_results_class, self.results_mask_classes[model_idt], self.random_results_task)
                    self.loggers[model_idt].write(vars(self.args))
