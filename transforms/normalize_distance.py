import torch
import torch.nn.functional as F

from utils.sample import allDistances


class NormalizeDistance(object):
    r"""Permutes axes so that the variances along axes are ascending, i.e. fixes up-vector.
    This assumes the shapes are aligned to one of the axes.

    Args:
        normalize_scale (bool, optional): if set to :obj:`True`, normalizes the scale of the shape
            such that the longest axis is in the range [0, 1].
            Should only be set after other precomputation steps have been performed.
    """

    def __init__(self, normalize_scale=True):
        self.normalize_scale = normalize_scale
        return

    def __call__(self, data):
        data.pos = data.pos / allDistances(data.pos).max()
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
