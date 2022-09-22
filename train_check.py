# File reading and progressbar
import os.path as osp
import progressbar

# PyTorch and PyTorch Geometric dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import zeros

# Harmonic Surface Networks components
# Layers
from nn import (HarmonicConv, HarmonicResNetBlock,
                ParallelTransportPool, ParallelTransportUnpool,
                ComplexLin, ComplexNonLin)
from transforms.normalize_distance import NormalizeDistance
# Utility functions
from utils.harmonic import magnitudes
# Rotated MNIST dataset
from datasets import ShapeSeg
# Transforms
from transforms import (HarmonicPrecomp, VectorHeatCloud, VectorHeat, MultiscaleRadiusGraph,
                        ScaleMask, FilterNeighbours, NormalizeArea, NormalizeAxes, Subsample)
from visualize import show_points, show_connection, show_weight


# Maximum rotation order for streams
max_order = 1

# Number of rings in the radial profile
n_rings = 6

# Number of filters per block
nf = [16, 32]

# Ratios used for pooling
ratios = [1, 0.25]

# Radius of convolution for each scale
radii = [0.2, 0.4]

# Number of datasets per batch
batch_size = 1

# Number of classes for segmentation
n_classes = 8


# 1. Provide a path to load and store the dataset.
# Make sure that you have created a folder 'data' somewhere
# and that you have downloaded and moved the raw datasets there
path = osp.join('data', 'ShapeSeg')

# 2. Define transformations to be performed on the dataset:
# Transformation that computes a multi-scale radius graph and precomputes the logarithmic map.
pre_transform = T.Compose((
    NormalizeDistance(),
    MultiscaleRadiusGraph(ratios, radii, loop=True,
                          flow='target_to_source', sample_n=100),
    VectorHeat(),
    Subsample(),
))
# Apply a random scale and random rotation to each shape
transform = T.Compose((
    T.RandomScale((0.85, 1.15)),
    T.RandomRotate(45, axis=0),
    T.RandomRotate(45, axis=1),
    T.RandomRotate(45, axis=2))
)

# Transformations that masks the edges and vertices per scale and precomputes convolution components.
scale0_transform = T.Compose((
    ScaleMask(0),
    FilterNeighbours(radii[0]),
    HarmonicPrecomp(n_rings, max_order, max_r=radii[0]))
)
scale1_transform = T.Compose((
    ScaleMask(1),
    FilterNeighbours(radii[1]),
    HarmonicPrecomp(n_rings, max_order, max_r=radii[1]))
)

# 3. Assign and load the datasets.
test_dataset = ShapeSeg(path, False, pre_transform=pre_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
train_dataset = ShapeSeg(
    path, True, pre_transform=pre_transform, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lin0 = nn.Linear(3, nf[0])

        # Stack 1
        self.resnet_block11 = HarmonicResNetBlock(
            nf[0], nf[0], max_order, n_rings, prev_order=0)
        self.resnet_block12 = HarmonicResNetBlock(
            nf[0], nf[0], max_order, n_rings)

        # Pool
        self.pool = ParallelTransportPool(1, scale1_transform)

        # Stack 2
        self.resnet_block21 = HarmonicResNetBlock(
            nf[0], nf[1], max_order, n_rings)
        self.resnet_block22 = HarmonicResNetBlock(
            nf[1], nf[1], max_order, n_rings)

        # Stack 3
        self.resnet_block31 = HarmonicResNetBlock(
            nf[1], nf[1], max_order, n_rings)
        self.resnet_block32 = HarmonicResNetBlock(
            nf[1], nf[1], max_order, n_rings)

        # Unpool
        self.unpool = ParallelTransportUnpool(from_lvl=1)

        # Stack 4
        self.resnet_block41 = HarmonicResNetBlock(
            nf[1] + nf[0], nf[0], max_order, n_rings)
        self.resnet_block42 = HarmonicResNetBlock(
            nf[0], nf[0], max_order, n_rings)

        # Final Harmonic Convolution
        # We set offset to False,
        # because we will only use the radial component of the features after this
        self.conv_final = HarmonicConv(
            nf[0], n_classes, max_order, n_rings, offset=False)

        self.bias = nn.Parameter(torch.Tensor(n_classes))
        zeros(self.bias)

    def forward(self, data):
        x = data.pos
        # Linear transformation from input positions to nf[0] features
        x = F.relu(self.lin0(x))

        # Convert input features into complex numbers
        x = torch.stack((x, torch.zeros_like(x)), dim=-1).unsqueeze(1)

        # Stack 1
        # Select only the edges and precomputed components of the first scale
        data_scale0 = scale0_transform(data)
        attributes = (data_scale0.edge_index,
                      data_scale0.precomp, data_scale0.connection)
        x = self.resnet_block11(x, *attributes)
        x_prepool = self.resnet_block12(x, *attributes)

        # Pooling
        # Apply parallel transport pooling
        x, data, data_pooled = self.pool(x_prepool, data)

        # Stack 2
        # Store edge_index and precomputed components of the second scale
        attributes_pooled = (data_pooled.edge_index,
                             data_pooled.precomp, data_pooled.connection)
        x = self.resnet_block21(x, *attributes_pooled)
        x = self.resnet_block22(x, *attributes_pooled)

        # Stack 3
        x = self.resnet_block31(x, *attributes_pooled)
        x = self.resnet_block32(x, *attributes_pooled)

        # Unpooling
        x = self.unpool(x, data)
        # Concatenate pre-pooling x with post-pooling x
        x = torch.cat((x, x_prepool), dim=2)

        # Stack 3
        x = self.resnet_block41(x, *attributes)
        x = self.resnet_block42(x, *attributes)

        x = self.conv_final(x, *attributes)

        # Take radial component from features and sum streams
        x = magnitudes(x, keepdim=False)
        x = x.sum(dim=1)

        x = x + self.bias
        return F.log_softmax(x, dim=1)


# We want to train on a GPU. It'll take a long time on a CPU
device = torch.device('cuda')
# Move the network to the GPU
model = Net().to(device)
# Set up the ADAM optimizer with learning rate of 0.0076 (as used in H-Nets)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    # Set model to 'train' mode
    model.train()

    if epoch > 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    for data in progressbar.progressbar(train_loader):
        # Move training data to the GPU and optimize parameters
        optimizer.zero_grad()
        F.nll_loss(model(data.to(device)), data.y).backward()
        optimizer.step()


def test():
    # Set model to 'evaluation' mode
    model.eval()
    correct = 0
    total_num = 0
    for i, data in enumerate(test_loader):
        pred = model(data.to(device)).max(1)[1]
        correct += pred.eq(data.y).sum().item()
        total_num += data.y.size(0)
    return correct / total_num


print('Start training, may take a while...')
# Try with fewer epochs if you're in a timecrunch
for epoch in range(50):
    train(epoch)
    test_acc = test()
    print("Epoch {} - Test: {:06.4f}".format(epoch, test_acc))
