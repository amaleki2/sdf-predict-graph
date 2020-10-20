import torch
import torch.nn as nn
from network import GCNet, DeeperGCN, GCNet3, UNet_reg, UNet
from train import plot_results, train_model
from data2 import get_sdf_data_loader, get_sdf_data_loader_from_sdf_pixels
#from pytorch3d.loss import chamfer_distance

n_objects, batch_size, n_epoch = 100, 1, 500
loss_func = nn.L1Loss()
edge_weight = True
step_size, gamma = 150, 0.5
lr_0 = 0.001

radius = 0.1
data_folder = "data2/dataset_1/graph5/"
train_data = get_sdf_data_loader(n_objects, data_folder, batch_size, edge_weight=edge_weight)

# mesh_folder = "data2/dataset_5/mesh/"
# train_data = get_sdf_data_loader_from_sdf_pixels(n_objects, mesh_folder, batch_size)
model = GCNet(3, [128, 128, 128, 128], 1)
train_model(model, train_data, lr_0=lr_0, n_epoch=n_epoch, with_borderless_loss=True, step_size=step_size,
            gamma=gamma, radius=radius, with_eikonal_loss=False)
plot_results(model, train_data, plot_every=15, levels=[-0.2, 0, 0.2, 0.4], border=0.1)