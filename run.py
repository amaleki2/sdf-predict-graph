from network import *
from train import plot_results, train_model, plot_results_over_line
from data2 import get_sdf_data_loader, get_sdf_data_loader_from_sdf_pixels
import sys
assert len(sys.argv) >= 3
model_name, data_folder = sys.argv[1:3]

edge_weight = False
n_objects, batch_size, n_epoch = 100, 15, 1500
lr_0, step_size, gamma, radius = 0.001, 200, 0.6, 0.1
in_channels, hidden_channels, out_channels = 9, [32, 64, 128, 64, 32], 1

if "graph" in data_folder:
    train_data = get_sdf_data_loader(n_objects, data_folder, batch_size, edge_weight=edge_weight)
else:
    train_data = get_sdf_data_loader_from_sdf_pixels(n_objects, data_folder, batch_size)

if model_name == "gcn":
    model = UNet_gcn_conv(in_channels, hidden_channels, out_channels)
elif model_name == "gat":
    head = 1
    model = UNet_gat_conv(in_channels, hidden_channels, out_channels, heads=head)
elif model_name == "feast":
    head = 1
    model = UNet_feast_conv(in_channels, hidden_channels, out_channels, heads=head)
# elif model_name == "spline":
#     model = UNet_spline_conv(in_channels, hidden_channels, out_channels)
# elif model_name == "edge":
#     model = UNet_edge_conv(in_channels, hidden_channels, out_channels)
else:
    raise(ValueError("Error"))

save_name = model_name + "_" + data_folder
save_name = save_name.replace("/", "_")
train_model(model, train_data, lr_0=lr_0, n_epoch=n_epoch, with_borderless_loss=False, step_size=step_size,
            gamma=gamma, radius=radius, with_eikonal_loss=False, save_name=save_name, print_every=100)
# plot_results(model, train_data, ndata=5, levels=[-0.2, 0, 0.2, 0.4], border=0.1, save_name="test")
# plot_results_over_line(model, train_data, ndata=5, save_name="test")