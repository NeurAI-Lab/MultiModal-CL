# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import subprocess
from typing import Any, Dict

import numpy as np
import csv
from utils import create_if_not_exists
from utils.conf import base_path
from utils.metrics import *

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'csv_log', 'notes', 'load_best_args']


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == 'domain-il':
        mean_acc, _ = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
            mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)


class Logger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str, output_dir: str, experiment_id: str,
                 model_idt: str = '') -> None:
        self.accs = []
        self.fullaccs = []
        if setting_str in ['class-il', 'multimodal-class-il']:
            self.accs_mask_classes = []
            self.fullaccs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.output_dir = output_dir
        self.experiment_id = experiment_id
        self.model_idt = model_idt
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None

    def dump(self):
        dic = {
            'accs': self.accs,
            'fullaccs': self.fullaccs,
            'fwt': self.fwt,
            'bwt': self.bwt,
            'forgetting': self.forgetting,
            'fwt_mask_classes': self.fwt_mask_classes,
            'bwt_mask_classes': self.bwt_mask_classes,
            'forgetting_mask_classes': self.forgetting_mask_classes,
        }
        if self.setting == 'class-il':
            dic['accs_mask_classes'] = self.accs_mask_classes
            dic['fullaccs_mask_classes'] = self.fullaccs_mask_classes

        return dic

    def load(self, dic):
        self.accs = dic['accs']
        self.fullaccs = dic['fullaccs']
        self.fwt = dic['fwt']
        self.bwt = dic['bwt']
        self.forgetting = dic['forgetting']
        self.fwt_mask_classes = dic['fwt_mask_classes']
        self.bwt_mask_classes = dic['bwt_mask_classes']
        self.forgetting_mask_classes = dic['forgetting_mask_classes']
        if self.setting == 'class-il':
            self.accs_mask_classes = dic['accs_mask_classes']
            self.fullaccs_mask_classes = dic['fullaccs_mask_classes']

    def rewind(self, num):
        self.accs = self.accs[:-num]
        self.fullaccs = self.fullaccs[:-num]
        with suppress(BaseException):
            self.fwt = self.fwt[:-num]
            self.bwt = self.bwt[:-num]
            self.forgetting = self.forgetting[:-num]
            self.fwt_mask_classes = self.fwt_mask_classes[:-num]
            self.bwt_mask_classes = self.bwt_mask_classes[:-num]
            self.forgetting_mask_classes = self.forgetting_mask_classes[:-num]

        if self.setting == 'class-il':
            self.accs_mask_classes = self.accs_mask_classes[:-num]
            self.fullaccs_mask_classes = self.fullaccs_mask_classes[:-num]

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        self.fwt = forward_transfer(results, accs)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(results_mask_classes, accs_mask_classes)

    def add_bwt(self, results, results_mask_classes):
        self.bwt = backward_transfer(results)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)

    def add_forgetting(self, results, results_mask_classes):
        self.forgetting = forgetting(results)
        self.forgetting_mask_classes = forgetting(results_mask_classes)

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        elif self.setting in ['domain-il', 'multimodal-domain-il']:
            mean_acc, _ = mean_acc
            self.accs.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)

    def log_fullacc(self, accs):
        if self.setting in ['class-il', 'multimodal-class-il']:
            acc_class_il, acc_task_il = accs
            self.fullaccs.append(acc_class_il)
            self.fullaccs_mask_classes.append(acc_task_il)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        wrargs = args.copy()
        columns = list(args.keys())

        new_cols = []
        for i, acc in enumerate(self.accs):
            wrargs['accmean_task' + str(i + 1)] = acc
            new_cols.append('accmean_task' + str(i + 1))

        for i, fa in enumerate(self.fullaccs):
            for j, acc in enumerate(fa):
                wrargs['accuracy_' + str(j + 1) + '_task' + str(i + 1)] = acc
                new_cols.append('accuracy_' + str(j + 1) + '_task' + str(i + 1))

        # new_cols.append("git hash")
        # wrargs["git hash"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

        wrargs['forward_transfer'] = self.fwt
        new_cols.append('forward_transfer')

        wrargs['backward_transfer'] = self.bwt
        new_cols.append('backward_transfer')

        wrargs['forgetting'] = self.forgetting
        new_cols.append('forgetting')

        columns = new_cols + columns
        logs_dir = os.path.join(self.output_dir, "results", self.setting,
                                self.dataset, self.model, self.experiment_id)
        os.makedirs(logs_dir, exist_ok=True)

        write_headers = False
        path = os.path.join(logs_dir, f"logs{self.model_idt}.csv")

        # Log task performance
        if self.setting != "general-continual":
            task_perf_path = os.path.join(logs_dir, f"task_performance{self.model_idt}.txt")
            n_tasks = len(self.fullaccs)
            results_array = np.zeros((n_tasks, n_tasks))
            print('Task Performance')
            for i in range(n_tasks):
                for j in range(n_tasks):
                    if i >= j:
                        results_array[i, j] = self.fullaccs[i][j]
            print(results_array)
            np.savetxt(task_perf_path, results_array, fmt='%.2f')

        if not os.path.exists(path):
            write_headers = True

        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(wrargs)

        if self.setting in ['class-il', 'multimodal-class-il']:
            logs_dir = os.path.join(self.output_dir, "results", "task-il",
                                    self.dataset, self.model, self.experiment_id)
            os.makedirs(logs_dir, exist_ok=True)

            for i, acc in enumerate(self.accs_mask_classes):
                wrargs['accmean_task' + str(i + 1)] = acc

            for i, fa in enumerate(self.fullaccs_mask_classes):
                for j, acc in enumerate(fa):
                    wrargs['accuracy_' + str(j + 1) + '_task' + str(i + 1)] = acc

            wrargs['forward_transfer'] = self.fwt_mask_classes
            wrargs['backward_transfer'] = self.bwt_mask_classes
            wrargs['forgetting'] = self.forgetting_mask_classes

            path = os.path.join(logs_dir, f"logs{self.model_idt}.csv")

            if not os.path.exists(path):
                write_headers = True
            with open(path, 'a') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                if write_headers:
                    writer.writeheader()
                writer.writerow(wrargs)
