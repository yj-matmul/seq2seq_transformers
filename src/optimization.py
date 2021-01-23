import math
import sys

import torch
from torch.optim import Optimizer


class LRSchedule():
    def __init__(self, warmup=0.002, t_total=-1):
        '''
        :param warmup: what fraction of t_total steps will be used for linear warmup
        :param t_total: how many training steps updates are planned
        '''
        super(LRSchedule, self).__init__()
        warmup = max(warmup, 0.)
        self.warmup = float(warmup)
        self.t_total = float(t_total)

    def get_lr(self, step):
        if self.t_total < 0:
            return 1
        progress = float(step) / self.t_total
        ret = self.get_lr_(progress)
        return ret

    def get_lr_(self, progress):
        return 1


class WarmupLinearSchedule(LRSchedule):
    warn_t_total = True
    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)



