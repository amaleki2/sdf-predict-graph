import torch
import torch.nn as nn
from network import GCNet, DeeperGCN, GCNet2
from train import plot_results, train_model
from data2 import get_sdf_data_loader

n_objects, batch_size, n_epoch = 100, 1, 1000
in_channels, out_channels = 3, 1
hidden_channels = [128, 128, 128, 128]
loss_func = nn.L1Loss()
edge_weight = False
step_size, gamma = 200, 0.5
lr_0 = 0.001

radius = 0.1
data_folder = "data2/dataset_1/graph7/"
train_data = get_sdf_data_loader(n_objects, data_folder, batch_size, edge_weight=edge_weight)
model = GCNet(in_channels, hidden_channels, out_channels)

# model = DeeperGCN(3, 128, 1, 4, 1)
train_model(model, train_data, lr_0=lr_0, n_epoch=n_epoch, with_borderless_loss=True, step_size=step_size,
            gamma=gamma, radius=radius, with_eikonal_loss=False)
plot_results(model, train_data, plot_every=5, levels=[-0.2, 0, 0.2, 0.4], border=0.1)