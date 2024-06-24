import torch
from torch import nn, optim
import torch.nn.functional as F


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.device = device
        self.a_temperature = torch.tensor([1.5], device=device, requires_grad=True)
        self.v_temperature = torch.tensor([1.5], device=device, requires_grad=True)
        self.av_temperature = torch.tensor([1.5], device=device, requires_grad=True)

    def forward(self, audio, video):
        a_logits, v_logits, av_logits = self.model(audio, video)
        return self.temperature_scale(a_logits, v_logits, av_logits)

    def temperature_scale(self, a_logits, v_logits, av_logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        a_temperature = self.a_temperature.unsqueeze(1).expand(a_logits.size(0), a_logits.size(1))
        v_temperature = self.v_temperature.unsqueeze(1).expand(v_logits.size(0), v_logits.size(1))
        av_temperature = self.av_temperature.unsqueeze(1).expand(av_logits.size(0), av_logits.size(1))
        return a_logits / a_temperature, v_logits / v_temperature, av_logits / av_temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, audio, video, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.to(self.device)
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        # First: collect all the logits and labels for the validation set
        a_logits_list = []
        v_logits_list = []
        av_logits_list = []
        labels_list = []

        with torch.no_grad():
            start_idx = 0
            batch_size = 32
            num_batch = len(audio) // batch_size

            for batch_idx in range(num_batch):
                end_idx = start_idx + batch_size
                # print(start_idx, end_idx)
                in_audio, in_video, in_label = audio[start_idx: end_idx], video[start_idx: end_idx], labels[start_idx: end_idx]

                a_out, v_out, av_out = self.model(in_audio.unsqueeze(1), in_video)
                a_logits_list.append(a_out)
                v_logits_list.append(v_out)
                av_logits_list.append(av_out)
                labels_list.append(in_label)
                start_idx = end_idx

            a_logits = torch.cat(a_logits_list).to(self.device)
            v_logits = torch.cat(v_logits_list).to(self.device)
            av_logits = torch.cat(av_logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # =====================================================================
        # Audio
        # =====================================================================
        # Calculate NLL and ECE before temperature scaling
        a_before_temperature_nll = nll_criterion(a_logits, labels).item()
        a_before_temperature_ece = ece_criterion(a_logits, labels).item()
        print('Audio: Before temperature - NLL: %.3f, ECE: %.3f' % (a_before_temperature_nll, a_before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.a_temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(a_logits, v_logits, av_logits)[0], labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(a_logits, v_logits, av_logits)[0], labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(a_logits, v_logits, av_logits)[0], labels).item()
        print('Audio: Optimal temperature: %.3f' % self.a_temperature.item())
        print('Audio: After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        # =====================================================================
        # Video
        # =====================================================================
        # Calculate NLL and ECE before temperature scaling
        v_before_temperature_nll = nll_criterion(v_logits, labels).item()
        v_before_temperature_ece = ece_criterion(v_logits, labels).item()
        print('Video: Before temperature - NLL: %.3f, ECE: %.3f' % (v_before_temperature_nll, v_before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.v_temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(a_logits, v_logits, av_logits)[1], labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(a_logits, v_logits, av_logits)[1], labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(a_logits, v_logits, av_logits)[1], labels).item()
        print('Video: Optimal temperature: %.3f' % self.v_temperature.item())
        print('Video: After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        # =====================================================================
        # Audio - Video
        # =====================================================================
        # Calculate NLL and ECE before temperature scaling
        av_before_temperature_nll = nll_criterion(av_logits, labels).item()
        av_before_temperature_ece = ece_criterion(av_logits, labels).item()
        print('Audio Video: Before temperature - NLL: %.3f, ECE: %.3f' % (av_before_temperature_nll, av_before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.av_temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(a_logits, v_logits, av_logits)[2], labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(a_logits, v_logits, av_logits)[2], labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(a_logits, v_logits, av_logits)[2], labels).item()
        print('Audio Video: Optimal temperature: %.3f' % self.av_temperature.item())
        print('Audio Video: After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
