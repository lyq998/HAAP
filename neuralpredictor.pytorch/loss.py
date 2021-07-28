from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import random


def select_positive(batch, dataset):
    AUG_LEN = 120
    org = batch['org_id']
    aug = batch['aug_id']
    positive_index_list = []
    for org_id, aug_id in zip(org, aug):
        # determine range
        positive_range = [AUG_LEN * org_id, AUG_LEN * (org_id + 1)]
        positive_index = random.choice(range(positive_range[0], positive_range[1]))
        while positive_index == aug_id:
            positive_index = random.choice(range(positive_range[0], positive_range[1]))
        positive_index_list.append(positive_index)

    positive_batch = {'num_vertices': [], 'adjacency': [], 'operations': []}
    for positive_index in positive_index_list:
        positive_info = dataset[positive_index]
        positive_batch['num_vertices'].append(positive_info['num_vertices'])
        positive_batch['adjacency'].append(positive_info['adjacency'])
        positive_batch['operations'].append(positive_info['operations'])

    positive_batch = {'num_vertices': Tensor(positive_batch['num_vertices']),
                      'adjacency': Tensor(positive_batch['adjacency']),
                      'operations': Tensor(positive_batch['operations'])}
    return positive_batch


def select_negative(batch, dataset):
    AUG_LEN = 120
    org = batch['org_id']
    negative_index_list = []
    for org_id in org:
        # determine range
        negative_range = [AUG_LEN * org_id, AUG_LEN * (org_id + 1)]
        negative_index = random.choice(list(range(0, negative_range[0])) + list(range(negative_range[1], len(dataset))))
        negative_index_list.append(negative_index)

    negative_batch = {'num_vertices': [], 'adjacency': [], 'operations': []}
    for negative_index in negative_index_list:
        negative_info = dataset[negative_index]
        negative_batch['num_vertices'].append(negative_info['num_vertices'])
        negative_batch['adjacency'].append(negative_info['adjacency'])
        negative_batch['operations'].append(negative_info['operations'])

    negative_batch = {'num_vertices': Tensor(negative_batch['num_vertices']),
                      'adjacency': Tensor(negative_batch['adjacency']),
                      'operations': Tensor(negative_batch['operations'])}
    return negative_batch


class MSELoss_and_TripletLoss(nn.Module):
    def __init__(self, alpha: float = 0.2, margin: float = 1.0, p: float = 2., eps: float = 1e-6, swap: bool = False,
                 reduction: str = 'mean'):
        super(MSELoss_and_TripletLoss, self).__init__()
        self.loss_name = 'MSELoss_and_TripletLoss'
        self.alpha = alpha
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        MSE = F.mse_loss(input, target, reduction=self.reduction)
        Triplet = F.triplet_margin_loss(input.reshape(-1, 1), positive.reshape(-1, 1), negative.reshape(-1, 1),
                                        margin=self.margin, p=self.p, eps=self.eps, swap=self.swap,
                                        reduction=self.reduction)
        return MSE + self.alpha * Triplet
