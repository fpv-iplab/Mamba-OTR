# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_criterion']

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os.path as osp
from functools import partial

from rekognition_online_action_detection.utils.registry import Registry

CRITERIONS = Registry()


@CRITERIONS.register('BCE')
class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(BinaryCrossEntropyLoss, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        return self.criterion(input, target)


@CRITERIONS.register('SCE')
class SingleCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100):
        super(SingleCrossEntropyLoss, self).__init__()

        self.criterion = nn.CrossEntropyLoss(
            reduction=reduction, ignore_index=ignore_index)

    def forward(self, input, target):
        return self.criterion(input, target)


@CRITERIONS.register('MCE')
class MultipCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100, tk_only=[]):
        super(MultipCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * logsoftmax(input[:, notice_index]), dim=1)
            if (target[:, self.ignore_index] != 1).sum().item() == 0:
                return torch.mean(torch.zeros_like(output))
            if self.reduction == 'mean':
                return torch.mean(output[target[:, self.ignore_index] != 1])
            elif self.reduction == 'sum':
                return torch.sum(output[target[:, self.ignore_index] != 1])
            else:
                return output[target[:, self.ignore_index] != 1]
        else:
            output = torch.sum(-target * logsoftmax(input), dim=1)

            if self.reduction == 'mean':
                return torch.mean(output)
            elif self.reduction == 'sum':
                return torch.sum(output)
            else:
                return output


@CRITERIONS.register('MCE_EQL')
class MultipCrossEntropyEqualizedLoss(nn.Module):

    def __init__(self, gamma=0.95, lambda_=3e-3, reduction='mean', ignore_index=-100,
                 anno_path='external/rulstm/RULSTM/data/ek55/', tk_only=[]):
        super(MultipCrossEntropyEqualizedLoss, self).__init__()

        # get label distribution
        segment_list = pd.read_csv(osp.join(anno_path, 'training.csv'),
                                   names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                                   skipinitialspace=True)
        numel = max(segment_list['action']) + 1 if len(tk_only) == 0 else len(tk_only) - 1
        freq_info = np.zeros((numel,))
        assert ignore_index == 0
        for segment in segment_list.iterrows():
            if len(tk_only) > 0:
                if segment[1]['action'] in tk_only:
                    freq_info[segment[1]['action'] % len(tk_only)] += 1.
            else:
                freq_info = freq_info / freq_info.sum()
        self.freq_info = torch.FloatTensor(freq_info)

        self.gamma = gamma
        self.lambda_ = lambda_
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)


        bg_target = target[:, self.ignore_index]
        notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
        input = input[:, notice_index]
        target = target[:, notice_index]

        weight = input.new_zeros(len(notice_index))
        weight[self.freq_info < self.lambda_] = 1.
        weight = weight.view(1, -1).repeat(input.shape[0], 1)
        
        eql_w = 1 - (torch.rand_like(target) < self.gamma) * weight * (1 - target)
        input = torch.log(eql_w + 1e-8) + input

        output = torch.sum(-target * logsoftmax(input), dim=1)
        if (bg_target != 1).sum().item() == 0:
            return torch.mean(torch.zeros_like(output))
        if self.reduction == 'mean':
            return torch.mean(output[bg_target != 1])
        elif self.reduction == 'sum':
            return torch.sum(output[bg_target != 1])
        else:
            return output[bg_target != 1]


@CRITERIONS.register('EQLv2')
class EQLv2Loss(nn.Module):

    def __init__(self, gamma=12, mu=0.8, alpha=4.0, reduction='mean', ignore_index=-100,
                 num_classes=3806):
        super(EQLv2Loss, self).__init__()
        self.num_classes = num_classes

        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.ignore_index = ignore_index

        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

    def forward(self, input, target):
        self.n_i, self.n_c = input.size()

        pos_w, neg_w = self.get_weight(input)

        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(input.detach(), target.detach(), weight.detach())

        return cls_loss

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)[1:]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[1:]

        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

    def get_weight(self, cls_score):
        if self._pos_grad is None:
            self._pos_grad = cls_score.new_zeros(self.num_classes)
            self._neg_grad = cls_score.new_zeros(self.num_classes)
            neg_w = cls_score.new_zeros((self.n_i, self.n_c))
            pos_w = cls_score.new_zeros((self.n_i, self.n_c))
        else:
            neg_w = torch.cat([cls_score.new_ones(1), self.map_func(self.pos_neg)])
            pos_w = 1 + self.alpha * (1 - neg_w)
            neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
            pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w



@CRITERIONS.register('FOCAL')
class FocalLoss(nn.Module):
    def __init__(self, 
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reg_lambda: float = 0.25,
                 regularization='entropy', 
                 window_size=5,
                 reduction='mean',
                 ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.reg_lambda = reg_lambda
        self.window_size = window_size
        self.ignore_index = ignore_index
        self.regularization = regularization


    def __entropy(self, p: torch.Tensor) -> torch.Tensor:
        h = -torch.sum(p * torch.log(p + 1e-6), dim=1)
        return h.mean()


    # def __sliding_window(self, p: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    #     _, seq_len = p.shape
    #     penalty = torch.zeros_like(p)

    #     for t in range(seq_len):
    #         start = max(0, t - self.window_size // 2)
    #         end = min(seq_len, t + self.window_size // 2 + 1)
    #         penalty[:, t] = torch.sum(p[:, start:end], dim=1) - p[:, t]

    #     return penalty.mean()


    def __sliding_window(self, p: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _, seq_len = p.shape
        penalty = torch.zeros_like(p)

        for t in range(seq_len):
            if targets[:, t].sum() > 0:
                start = max(0, t - self.window_size // 2)
                end = min(seq_len, t + self.window_size // 2 + 1)
                penalty[:, t] = torch.sum(p[:, start:end], dim=1) - p[:, t]

        return penalty.mean()


    def forward(self, 
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Taken from
        https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
        # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = 0.25.
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        inputs = inputs.float()
        targets = targets.float()

        notice_index = [i for i in range(targets.shape[-1]) if i != self.ignore_index]
        inputs = inputs[:, notice_index]
        targets = targets[:, notice_index]

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)

        h = 0
        loss = None
        if self.regularization == 'entropy':
            h = self.__entropy(p)
        elif self.regularization == 'sliding_window':
            h = self.__sliding_window(p, targets)
        else:
            raise ValueError(f"Regularization method {self.regularization} not supported")

        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss + self.reg_lambda * h


@CRITERIONS.register('MCE_SEESAW')
class MultipCrossEntropySeesawLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100,
                 p=0.8, q=2.0, eps=1e-2,
                 anno_path='external/rulstm/RULSTM/data/ek55/'):
        super(MultipCrossEntropySeesawLoss, self).__init__()

        # get label distribution
        segment_list = pd.read_csv(osp.join(anno_path, 'training.csv'),
                                   names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                                   skipinitialspace=True)
        self.num_classes = max(segment_list['action']) + 1
        self.cum_samples = torch.zeros(self.num_classes, dtype=torch.float)

        self.p = p
        self.q = q
        self.eps = eps

        self.reduction = reduction
        assert ignore_index == 0
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        self.cum_samples = self.cum_samples.to(input.device)

        bg_target = target[:, self.ignore_index]
        notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
        input = input[:, notice_index]
        target = target[:, notice_index]

        # accumulate the samples for each category
        self.cum_samples += target.sum(0)
        
        seesaw_weights = input.new_ones(target.size())

        if self.p > 0:
            sample_ratio_matrix = self.cum_samples[None, :].clamp(min=1) / self.cum_samples[:, None].clamp(min=1)
            index = (sample_ratio_matrix < 1.0).float()
            sample_weights = sample_ratio_matrix.pow(self.p) * index + (1 - index)
            for ni in range(input.shape[0]):
                true_target = target[ni, :].nonzero(as_tuple=True)[0]
                ## TODO: study mean/min/max/sum
                if len(true_target) > 0:
                    mitigation_factor = torch.min(sample_weights[true_target, :], dim=0).values
                    seesaw_weights[ni, :] = seesaw_weights[ni, :] * mitigation_factor
        if self.q > 0:
            scores = F.softmax(input.detach(), dim=1)
            self_scores = input.new_ones([input.shape[0]])
            for ni in range(input.shape[0]):
                true_target = target[ni, :].nonzero(as_tuple=True)[0]
                ## TODO: study mean/min/max/sum
                if len(true_target) > 0:
                    self_scores[ni] = torch.min(scores[ni, true_target])
            score_matrix = scores / self_scores[:, None].clamp(min=self.eps)
            index = (score_matrix > 1.0).float()
            compensation_factor = score_matrix.pow(self.q) * index + (1 - index)
            seesaw_weights = seesaw_weights * compensation_factor

        input = input + (seesaw_weights.log() * (1 - target))

        output = torch.sum(-target * logsoftmax(input), dim=1)
        if (bg_target != 1).sum().item() == 0:
            return torch.mean(torch.zeros_like(output))
        if self.reduction == 'mean':
            return torch.mean(output[bg_target != 1])
        elif self.reduction == 'sum':
            return torch.sum(output[bg_target != 1])
        else:
            return output[bg_target != 1]



def build_criterion(cfg, device=None):
    criterion = {}
    for name, params in cfg.MODEL.CRITERIONS:
        if name in CRITERIONS:
            if 'ignore_index' not in params:
                params['ignore_index'] = cfg.DATA.IGNORE_INDEX
            if name == "MCE_EQL":
                params["tk_only"] = cfg.DATA.TK_IDXS if cfg.DATA.TK_ONLY else []
            criterion[name] = CRITERIONS[name](**params).to(device)
        else:
            raise RuntimeError('Unknown criterion: {}'.format(name))
    return criterion
