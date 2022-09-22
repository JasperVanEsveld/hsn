import torch
from math import pi as PI

# Convert edge atributes to polar coordinates
def polar(edge_attr):
  if not torch.is_tensor(edge_attr):
    edge_attr = torch.tensor(edge_attr)
  r = edge_attr.norm(dim=1)
  theta = torch.atan2(edge_attr[:, 1], edge_attr[:, 0])
  theta = theta + (theta < 0).type_as(theta) * (2 * PI)
  theta = theta / (2 * PI)
  return torch.stack((r, theta), dim=-1)