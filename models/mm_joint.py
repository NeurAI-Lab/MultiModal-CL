# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
from datasets.utils.validation import MMValidationDataset
from torch.optim import SGD
from torchvision import transforms

from models.utils.av_continual_model import AVContinualModel
from utils.args import *
from utils.status import progress_bar


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MMJoint(AVContinualModel):
    NAME = 'mm_joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual', 'multi-modal']

    def __init__(self, backbone, loss, args, transform):
        super(MMJoint, self).__init__(backbone, loss, args, transform)
        self.old_audio = []
        self.old_video = []
        self.old_labels = []
        self.current_task = 0
        self.addit_modalities = ['audio', 'video']

    def end_task(self, dataset):

        self.old_audio.append(dataset.train_loader.dataset.audio)
        self.old_video.append(dataset.train_loader.dataset.video)
        self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
        self.current_task += 1

        # # for non-incremental joint training
        if len(dataset.test_loaders) != dataset.N_TASKS:
            return

        # reinit network
        self.net = dataset.get_backbone()
        self.net.to(self.device)
        self.net.train()
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)

        # prepare dataloader
        all_audio, all_video, all_labels = None, None, None
        for i in range(len(self.old_video)):
            if all_video is None:
                all_audio = self.old_audio[i]
                all_video = self.old_video[i]
                all_labels = self.old_labels[i]
            else:
                all_audio = np.concatenate([all_audio, self.old_audio[i]])
                all_video = np.concatenate([all_video, self.old_video[i]])
                all_labels = np.concatenate([all_labels, self.old_labels[i]])

        temp_dataset = MMValidationDataset(all_audio, all_video, all_labels, num_video_frames=self.args.num_video_frames)
        loader = torch.utils.data.DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=True)

        # train
        for e in range(self.args.n_epochs):
            for i, batch in enumerate(loader):
                (audio, video), labels = batch
                audio, video, labels = audio.to(self.device), video.to(self.device), labels.to(self.device)

                self.opt.zero_grad()
                a, v, outputs = self.net(audio.unsqueeze(1), video)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.opt.step()
                progress_bar(i, len(loader), e, 'J', loss.item())

        self.eval_addit_modalities(dataset)

    def observe(self, inputs, labels, not_aug_inputs):
        return 0

    def begin_task(self, dataset):

        # Init Logs at the beginning of training
        if self.task == 0:
            print('Initializing logs for additional Modalities')
            self.init_loggers(dataset)
            self.eval_before_training(dataset)
