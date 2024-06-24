from models.utils.av_continual_model import AVContinualModel
from models.utils.temperature_calibration import ModelWithTemperature
from utils.args import *
from utils.mm_buffer import Buffer
from models.utils.relational_distillation import RKD
import torch
from torch import nn
from torch.nn import functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='MultiModal Continual learning via Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # Consistency Regularization Weight
    parser.add_argument('--reg_weight', type=float, default=0.01)
    parser.add_argument('--cons_weight', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--feature_alignment_weight', type=float, default=25)
    parser.add_argument('--angle_weight', type=float, default=50)
    parser.add_argument('--cons_weight', type=float, default=1)
    parser.add_argument('--save_models', type=int, default=0)
    return parser


class MMCL(AVContinualModel):
    NAME = 'mmcl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual', 'multi-modal']

    def __init__(self, backbone, loss, args, transform):
        super(MMCL, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.addit_modalities = ['audio', 'video', 'calib_dynamic_av_weighted_ensemble']
        self.feature_alignment_loss = RKD(self.device, eval_dist_loss=True, eval_angle_loss=False)
        self.consistency_loss = nn.MSELoss(reduction='none')

    def observe(self, inputs, labels, not_aug_inputs):

        audio, video = inputs
        audio, video, labels = audio.to(self.device), video.to(self.device), labels.to(self.device)
        real_batch_size = audio.shape[0]

        loss = 0
        self.opt.zero_grad()

        # =====================================================================
        # Train on Current Task
        # =====================================================================
        a_out, v_out, av_out, feats = self.net(audio.unsqueeze(1), video, return_feat=True)
        av_loss = self.loss(av_out, labels)
        a_cl_loss = self.loss(a_out, labels)
        v_cl_loss = self.loss(v_out, labels)

        a_distance_loss, _ = self.feature_alignment_loss.eval_loss(a_out, v_out.detach())
        v_distance_loss, _ = self.feature_alignment_loss.eval_loss(v_out, a_out.detach())
        a_fa_loss = self.args.feature_alignment_weight * a_distance_loss
        v_fa_loss = self.args.feature_alignment_weight * v_distance_loss

        a_loss = a_cl_loss + a_fa_loss
        v_loss = v_cl_loss + v_fa_loss

        loss += av_loss + self.args.reg_weight * (a_loss + v_loss)

        # =====================================================================
        # Train on Buffer Samples
        # =====================================================================
        if not self.buffer.is_empty():
            buf_audio, buf_video, buf_labels, buf_av_logits, buf_a_logits, buf_v_logits = self.buffer.get_data(
                self.args.minibatch_size,
                audio_transform=None,
                video_transform=self.transform,
            )

            buf_a_out, buf_v_out, buf_av_out, buf_feats = self.net(buf_audio.unsqueeze(1), buf_video, return_feat=True)
            a_prob = F.softmax(buf_a_out, 1)
            v_prob = F.softmax(buf_v_out, 1)
            av_prob = F.softmax(buf_av_out, 1)

            # Dynamic Logit Selection
            buf_logits = torch.cat([buf_a_logits.unsqueeze(0), buf_v_logits.unsqueeze(0), buf_av_logits.unsqueeze(0)])
            buf_logits = buf_logits.permute((1, 0, 2))

            label_mask = F.one_hot(buf_labels, num_classes=buf_a_out.shape[-1]) > 0
            probs = torch.cat([a_prob[label_mask].unsqueeze(0), v_prob[label_mask].unsqueeze(0), av_prob[label_mask].unsqueeze(0)])

            sel_idx = probs.max(dim=0)[1]
            cons_logits = buf_logits[torch.arange(buf_logits.shape[0]), sel_idx]

            # Consistency Loss
            buf_l_av_cons = torch.mean(self.consistency_loss(buf_av_out, cons_logits))
            buf_l_a_cons = torch.mean(self.consistency_loss(buf_a_out, cons_logits))
            buf_l_v_cons = torch.mean(self.consistency_loss(buf_v_out, cons_logits))

            buf_l_cons = buf_l_av_cons + self.args.reg_weight * (buf_l_a_cons + buf_l_v_cons)
            loss += self.args.cons_weight * buf_l_cons

            # Classification Loss
            buf_av_loss = self.loss(buf_av_out, buf_labels)
            buf_a_cl_loss = self.loss(buf_a_out, buf_labels)
            buf_v_cl_loss = self.loss(buf_v_out, buf_labels)

            # Feature Alignment Loss
            buf_a_distance_loss, _ = self.feature_alignment_loss.eval_loss(buf_a_out, buf_v_out.detach())
            buf_v_distance_loss, _ = self.feature_alignment_loss.eval_loss(buf_v_out, buf_a_out.detach())
            buf_a_kl_loss = self.args.feature_alignment_weight * buf_a_distance_loss
            buf_v_kl_loss = self.args.feature_alignment_weight * buf_v_distance_loss

            buf_a_loss = buf_a_cl_loss + buf_a_kl_loss
            buf_v_loss = buf_v_cl_loss + buf_v_kl_loss

            loss += buf_av_loss + self.args.reg_weight * (buf_a_loss + buf_v_loss)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            audio=not_aug_inputs[0].to(self.device),
            video=not_aug_inputs[1].to(self.device),
            labels=labels[:real_batch_size],
            logits=av_out[:real_batch_size].detach(),
            a_logits=a_out[:real_batch_size].detach(),
            v_logits=v_out[:real_batch_size].detach(),
        )

        return loss.item()

    def begin_task(self, dataset):

        # Init Logs at the beginning of training
        if self.task == 0:
            print('Initializing logs for additional Modalities')
            self.init_loggers(dataset)

            self.eval_before_training(dataset)

    def end_task(self, dataset):

        # Calibrate the Model using the buffered samples
        self.calib_net = ModelWithTemperature(self.net, self.device)

        buf_audio, buf_video, buf_labels, _, _, _ = self.buffer.get_all_data(
            audio_transform=None,
            video_transform=self.transform,
        )

        self.calib_net.set_temperature(buf_audio, buf_video, buf_labels)

        self.task += 1
        self.iteration = 0

        if self.args.save_models:
            self.save_models(dataset)
