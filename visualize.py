import numpy as np
import polyscope as ps
import torch
from torch_geometric.utils import degree

ps.init()


def show_neighbours(index, points, sample_idx, edge_index, deg):
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    point_index = sample_idx[index]
    start = int(deg[0:index].sum())
    end = int(start + deg[index])
    ps.register_point_cloud("points", points)
    ps.register_point_cloud(
        "neighbours", points[sample_idx[edge_index[1, start:end]]], radius=0.01)
    ps.register_point_cloud(
        "selected", points[[point_index]], radius=0.011)
    ps.show()


def show_weight(index, points, edge_index, weight):
    deg = degree(edge_index[0]).cpu().numpy()
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(weight):
        weight = weight.cpu().numpy()
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    n = points.shape[0]
    start = int(deg[0:index].sum())
    end = int(start + deg[index])
    colors = np.ones((n, 3))
    colors[edge_index[1, start:end], 0] = weight[start:end]
    ps_cloud = ps.register_point_cloud("points", points)
    ps_cloud.add_color_quantity("weight", colors)
    ps.register_point_cloud(
        "selected", points[[index]], radius=0.01)
    ps.show()


def show_points(index, points, edge_index, edge_attr):
    deg = degree(edge_index[0]).cpu().numpy()
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(edge_attr):
        edge_attr = edge_attr.cpu().numpy()
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    n = points.shape[0]
    start = int(deg[0:index].sum())
    end = int(start + deg[index])
    colors = np.ones((n, 3))
    colors[edge_index[1, start:end], :2] = edge_attr[start:end]
    ps_cloud = ps.register_point_cloud("points", points)
    ps_cloud.add_color_quantity("attr", colors)
    ps.register_point_cloud(
        "selected", points[[index]], radius=0.01)
    ps.show()


def show_connection(index, points, edge_index, connection, deg):
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(connection):
        connection = connection.cpu().numpy()
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    n = points.shape[0]
    start = int(deg[0:index].sum())
    end = int(start + deg[index])
    ps_cloud = ps.register_point_cloud("points", points)
    neighbours = edge_index[1, start:end]
    transported = np.zeros((n, 3))
    x_comp = np.array([1, 0, 0]) * connection[start:end, 0, np.newaxis]
    y_comp = np.array([0, 1, 0]) * connection[start:end, 1, np.newaxis]
    transported[neighbours, :] = x_comp + y_comp
    ps_cloud.add_vector_quantity(
        "transport", transported)
    ps.register_point_cloud(
        "selected", points[[index]], radius=0.01)
    ps.show()


def show_cloud(index, points):
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    ps.register_point_cloud("points", points)
    ps.register_point_cloud(
        "selected", points[[index]], radius=0.01)
    ps.show()
