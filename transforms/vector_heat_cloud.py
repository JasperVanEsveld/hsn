import os.path as osp
import numpy as np
import vectorheat_pointcloud as pp3d
import torch
from math import pi as PI
from utils import polar
from torch_geometric.utils import degree


def map_index(edge_index, sample_idx):
    from_index = np.array([sample_idx[xi] for xi in edge_index[0]])
    to_index = np.array([sample_idx[xi] for xi in edge_index[1]])
    return np.vstack((from_index, to_index))


def precomp_all(points, edge_index, deg):
    solver = pp3d.PointCloudHeatSolver(points)
    data = solver.precompute(edge_index, deg)
    edge_attr = data[:, :2]
    transport = data[:, 2:]
    weight = solver.get_lumped_mass()
    weight = weight / weight.max()
    return edge_attr, transport, weight


class VectorHeatCloud(object):

    def __init__(self, cache_file=None):
        self.get_cache = cache_file is not None and osp.exists(cache_file)
        self.save_cache = cache_file is not None and not self.get_cache
        self.cache_file = cache_file

        if self.get_cache:
            self.connection, self.edge_attr, self.weight = torch.load(
                cache_file)

    def __call__(self, data):
        deg = degree(data.edge_index[0])
        mapped_index = map_index(data.edge_index, data.sample_idx)
        edge_attr, transport, weight = precomp_all(data.pos, mapped_index, deg)
        data.connection = torch.tensor(transport)
        data.weight = torch.tensor(weight, dtype=torch.float)
        data.weight = data.weight[mapped_index[1]].view(-1)
        data.edge_attr = polar(edge_attr).float()

        if self.save_cache:
            with open(self.cache_file, 'wb') as f:
                torch.save((data.connection, data.edge_attr, data.weight), f)
            self.save_cache = False
            self.get_cache = True
            self.connection, self.edge_attr, self.weight = data.connection, data.edge_attr, data.weight
        return data

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
