import torch
import torch_cluster
from torch_geometric.nn import radius, fps
import numpy as np
from torch_geometric.utils import degree as calcDegree


def allDistances(points):
    return torch.cdist(points, points, p=2)


def distances(points, indices):
    positions = points[:, :3]
    samples_points = points[indices, :3]
    return torch.cdist(positions, samples_points, p=2)


def within_sphere(source_point, dist, radius):
    return (dist[source_point] <= radius).nonzero().flatten()


def cluster_radius(points, radius):
    return torch_cluster.radius(points, points, radius)

# Gets neighbours for each points
# First subsample with FPS and than take all points in radius around a point


def sample_fps_radius(points, ratio, r):
    if not torch.is_tensor(points):
        points = torch.tensor(points)
    fps_indices = fps(points, batch=None, ratio=ratio)
    edge_index = radius(points[fps_indices], points, r)
    edge_index[1] = fps_indices[edge_index[1]]
    degree = calcDegree(edge_index[0])
    return (edge_index, degree)
