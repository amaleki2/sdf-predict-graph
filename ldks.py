import torch
import torch.nn as nn
from network import GCNet
from train import plot_results
from data2 import get_sdf_data_loader

n_objects, batch_size = 100, 1
in_channels, out_channels = 3, 1
hidden_channels = [128, 128, 128, 128, 128]
loss_func = nn.L1Loss()
edge_weight = True

radius = 0.1
data_folder = "data2/dataset_1/graph11/"
train_data = get_sdf_data_loader(n_objects, data_folder, batch_size, edge_weight=edge_weight)
model = GCNet(in_channels, hidden_channels, out_channels)
#plot_results(model, train_data, plot_every=5, levels=[-0.2, 0, 0.2, 0.4], border=0.1)

from network import DeeperGCN
device = "cpu"
model = DeeperGCN(3, 64, 1, 4, 1).to(device)
for d in train_data:
    d = d.to(device)
    u = model(d)
    print(u)
