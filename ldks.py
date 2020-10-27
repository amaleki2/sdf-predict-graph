import torch
import torch.nn as nn
from network import GCNet, UNet_feast_conv
from train import plot_results, train_model, plot_results_over_line
from data2 import get_sdf_data_loader, get_sdf_data_loader_from_sdf_pixels


n_objects, batch_size, n_epoch = 100, 1, 50
loss_func = nn.L1Loss()
edge_weight = True
step_size, gamma = 150, 0.5
lr_0 = 0.001

radius = 0.1
data_folder = "data2/dataset_1/graph5/"
train_data = get_sdf_data_loader(n_objects, data_folder, batch_size, edge_weight=edge_weight)

# mesh_folder = "data2/dataset_5/mesh/"
# train_data = get_sdf_data_loader_from_sdf_pixels(n_objects, mesh_folder, batch_size)
model = UNet_feast_conv(3, [10, 20, 30, 40], 1)
train_model(model, train_data, lr_0=lr_0, n_epoch=n_epoch, with_borderless_loss=True, step_size=step_size,
            gamma=gamma, radius=radius, with_eikonal_loss=True, save_name="test")
#plot_results(model, train_data, ndata=5, levels=[-0.2, 0, 0.2, 0.4], border=0.1, save_name="test")
plot_results_over_line(model, train_data, ndata=5, save_name="test")