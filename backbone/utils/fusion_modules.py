import torch
import torch.nn as nn
import torch.nn.functional as F


class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        output = out_x + out_y
        return out_x, out_y, output


class WeightedSumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(WeightedSumFusion, self).__init__()

        self.att_x = nn.Linear(input_dim, input_dim)
        self.att_y = nn.Linear(input_dim, input_dim)

        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):

        weight_x = torch.sigmoid(self.att_x(x))
        weight_y = torch.sigmoid(self.att_y(y))

        out_x = self.fc_x(weight_x * x)
        out_y = self.fc_y(weight_y * y)

        output = out_x + out_y
        return out_x, out_y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim * 2, output_dim)
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return out_x, out_y, output


class WeightedConcatFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(WeightedConcatFusion, self).__init__()
        self.att_x = nn.Linear(input_dim, input_dim)
        self.att_y = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim * 2, output_dim)
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        weight_x = F.sigmoid(self.att_x(x))
        weight_y = F.sigmoid(self.att_y(y))

        out_x = self.fc_x(x)
        out_y = self.fc_y(y)
        output = torch.cat((weight_x * x, weight_y * y), dim=1)
        output = self.fc_out(output)
        return out_x, out_y, output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, output_dim=100, x_film=True,  modalities='audio_video'):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * input_dim)
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.x_out = nn.Linear(input_dim, output_dim)
        self.y_out = nn.Linear(input_dim, output_dim)
        self.modalities = modalities
        self.x_film = x_film

    def forward(self, x, y, return_feat=False):

        if self.modalities == 'audio_video':
            if self.x_film:
                film = x
                to_be_film = y
            else:
                film = y
                to_be_film = x

            gamma, beta = torch.split(self.fc(film), self.dim, 1)

            av_feat = gamma * to_be_film + beta
            output = self.fc_out(av_feat)

        elif self.modalities == 'audio':
            output = self.fc_out(x)

        elif self.modalities == 'video':
            output = self.fc_out(y)

        out_x = self.x_out(x)
        out_y = self.y_out(y)

        if return_feat:
            return out_x, out_y, output, av_feat

        return out_x, out_y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True,  modalities='audio_video'):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_out = nn.Linear(dim, output_dim)
        self.y_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate
        self.modalities = modalities
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.modalities == 'audio_video':
            if self.x_gate:
                gate = self.sigmoid(out_x)
                output = self.fc_out(torch.mul(gate, out_y))
            else:
                gate = self.sigmoid(out_y)
                output = self.fc_out(torch.mul(out_x, gate))
        elif self.modalities == 'audio':
            output = self.fc_out(out_x)

        elif self.modalities == 'video':
            output = self.fc_out(out_y)

        out_x = self.x_out(out_x)
        out_y = self.y_out(out_y)

        return out_x, out_y, output
