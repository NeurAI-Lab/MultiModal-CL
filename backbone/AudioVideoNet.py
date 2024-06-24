import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.utils.av_backbone import resnet18
from backbone.utils.fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, WeightedSumFusion, SumFusion_v2, WeightedSumFusion_v2, WeightedConcatFusion


class AVClassifier(nn.Module):
    def __init__(self, n_classes, fusion, modalities_used, base_width=64):
        super(AVClassifier, self).__init__()

        if fusion == 'sum':
            self.fusion_module = SumFusion(input_dim=base_width * 8,  output_dim=n_classes)
        elif fusion == 'weighted_sum':
            self.fusion_module = WeightedSumFusion(input_dim=base_width * 8, output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(input_dim=base_width * 8, output_dim=n_classes)
        elif fusion == 'weighted_concat':
            self.fusion_module = WeightedConcatFusion(input_dim=base_width * 8, output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(input_dim=base_width * 8, output_dim=n_classes, x_film=True, modalities=modalities_used)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(input_dim=base_width * 8, output_dim=n_classes, x_gate=True, modalities=modalities_used)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio', base_width=base_width)
        self.visual_net = resnet18(modality='visual', base_width=base_width)

    def forward(self, audio, visual, return_feat=False):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a_feat = torch.flatten(a, 1)
        v_feat = torch.flatten(v, 1)

        if return_feat:
            a, v, out, av_feat = self.fusion_module(a_feat, v_feat, return_feat)
            dict_feat = {
                'audio': a_feat,
                'video': v_feat,
                'audio_video': av_feat,
            }
            return a, v, out, dict_feat

        a, v, out = self.fusion_module(a_feat, v_feat)

        return a, v, out
