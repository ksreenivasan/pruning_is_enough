import torch.nn as nn

LearnedBatchNorm = nn.BatchNorm2d


class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim, eps=1e-05, momentum=0.1):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False, eps=eps, momentum=momentum)


class AffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(AffineBatchNorm, self).__init__(dim, affine=True)

