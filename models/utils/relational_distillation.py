import torch


class RKD(object):
    """
    Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.
    relational knowledge distillation.
    arXiv preprint arXiv:1904.05068, 2019.
    """
    def __init__(self, device, eval_dist_loss=True, eval_angle_loss=False):
        super(RKD, self).__init__()
        self.device = device
        self.eval_dist_loss = eval_dist_loss
        self.eval_angle_loss = eval_angle_loss
        self.huber_loss = torch.nn.SmoothL1Loss()

    @staticmethod
    def distance_wise_potential(x):
        x_square = x.pow(2).sum(dim=-1)
        prod = torch.matmul(x, x.t())
        distance = torch.sqrt(
            torch.clamp(torch.unsqueeze(x_square, 1) + torch.unsqueeze(x_square, 0) - 2 * prod,
            min=1e-12))
        mu = torch.sum(distance) / torch.sum(
            torch.where(distance > 0., torch.ones_like(distance),
                        torch.zeros_like(distance)))

        return distance / (mu + 1e-8)

    @staticmethod
    def angle_wise_potential(x):
        e = torch.unsqueeze(x, 0) - torch.unsqueeze(x, 1)
        e_norm = torch.nn.functional.normalize(e, dim=2)
        return torch.matmul(e_norm, torch.transpose(e_norm, -1, -2))

    def eval_loss(self, source, target):

        # Flatten tensors
        source = source.reshape(source.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        # normalize
        source = torch.nn.functional.normalize(source, dim=1)
        target = torch.nn.functional.normalize(target, dim=1)

        distance_loss = torch.tensor([0.]).to(self.device)
        angle_loss = torch.tensor([0.]).to(self.device)

        if self.eval_dist_loss:
            distance_loss = self.huber_loss(
                self.distance_wise_potential(source), self.distance_wise_potential(target)
            )

        if self.eval_angle_loss:
            angle_loss = self.huber_loss(
                self.angle_wise_potential(source), self.angle_wise_potential(target)
            )

        return distance_loss, angle_loss
