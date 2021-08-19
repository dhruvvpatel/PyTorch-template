# :: dhruv :: #


import torch.nn.functional as F

def nll_loss(pred, gt):
    return F.nll_loss(pred, gt)
