# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.av_continual_model import AVContinualModel
from utils.args import *
from utils.mm_buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='MultiModal Continual learning via Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class MMEr(AVContinualModel):
    NAME = 'mm_er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual', 'multi-modal']

    def __init__(self, backbone, loss, args, transform):
        super(MMEr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        audio, video = inputs
        audio, video, labels = audio.to(self.device), video.to(self.device), labels.to(self.device)
        real_batch_size = audio.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_audio, buf_video, buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                audio_transform=None,
                video_transform=self.transform,
            )

            audio = torch.cat((audio, buf_audio))
            video = torch.cat((video, buf_video))
            labels = torch.cat((labels, buf_labels))

        _, _, outputs = self.net(audio.unsqueeze(1), video)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            audio=not_aug_inputs[0].to(self.device),
            video=not_aug_inputs[1].to(self.device),
            labels=labels[:real_batch_size]
        )

        return loss.item()

    def end_task(self, dataset):
        print('Saving Model')
        self.save_models(dataset)
