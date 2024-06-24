# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.utils.av_continual_model import AVContinualModel
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Multi Modal (Audio and Video) SGD')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class MMSgd(AVContinualModel):
    NAME = 'mm_sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual', 'multi-modal']

    def __init__(self, backbone, loss, args, transform):
        super(MMSgd, self).__init__(backbone, loss, args, transform)

    def observe(self, inputs, labels, not_aug_inputs):

        audio, video = inputs
        audio, video, labels = audio.to(self.device), video.to(self.device), labels.to(self.device)

        self.opt.zero_grad()
        a, v, outputs = self.net(audio.unsqueeze(1), video)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
