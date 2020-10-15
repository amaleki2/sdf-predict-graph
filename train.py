import meshio
import numpy as np
import matplotlib.pyplot as plt
from data2 import plot_mesh

import torch
import torch.nn as nn

l1_loss = nn.L1Loss
l2_loss = nn.MSELoss


def borderless_loss(pred, target, loss_func, data, radius):
    mask = torch.logical_and(torch.abs(data.x[:, 0]) < 1 - radius, torch.abs(data.x[:, 1]) < 1 - radius)
    loss = loss_func(reduction='none')(pred, target)
    loss_masked = loss[mask]
    loss_masked_reduced = torch.mean(loss_masked)
    return loss_masked_reduced


def eikonal_loss(pred, xy, device='cuda', retain_graph=True):
    pred.backward(gradient=torch.ones(pred.size()).to(device), retain_graph=retain_graph)
    dg = xy.grad[:, :2]
    dg_mag = torch.sqrt(torch.sum(dg * dg, dim=-1))
    eikonal_loss = l2_loss(dg_mag, torch.ones(dg_mag.size()).to(device))
    eikonal_loss.requires_grad = True
    return eikonal_loss


def train_model(model, train_data, lr_0=0.001, n_epoch=101, loss_func=l1_loss,
                with_borderless_loss=True, with_eikonal_loss=False,
                print_every=10, step_size=50, gamma=0.5, radius=0.1):

    print("training begins")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_0, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    running_loss_list = []
    for epoch in range(1, n_epoch):
        running_loss = 0
        for d in train_data:
            d = d.to(device)
            if with_eikonal_loss:
                d.x.requires_grad = True
                #Todo: make sure d.x does not participate in optimization.
            model.train()
            optimizer.zero_grad()
            pred = model(d)
            target = d.y
            if with_borderless_loss:
                loss = borderless_loss(pred, target, loss_func, d, radius)
            else:
                loss = l1_loss()(pred, target)

            loss += eikonal_loss(pred, d.x) if with_eikonal_loss else 0
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        running_loss_list.append(running_loss / len(train_data))
        if epoch % print_every == 0:
            print("epoch=%d, loss=%0.5e, lr=%0.5e" % (epoch, running_loss / len(train_data),
                                                      optimizer.param_groups[0]['lr']))
        if epoch % (10 * print_every) == 0:
            torch.save(model.state_dict(), "models/model" + ".pth")
            np.save("models/loss.npy", running_loss_list)
            #plot_results(model, train_data, plot_every=5)


def plot_results(model, data, plot_every=-1, levels=None, border=None):
    if plot_every == -1:
        p = 1
    else:
        p = plot_every / len(data)

    loss_history = np.load("models/loss.npy")
    plt.plot(loss_history)
    plt.yscale('log')

    device = 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load("models/model" + ".pth", map_location=device))
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(data):
            if np.random.random() > p:
                continue
            d = d.to(device=device)
            pred = model(d)
            pred += torch.sum(d.x[:, :2], dim=-1, keepdim=True)
            cells = d.face.numpy()
            points = d.x.numpy()
            points[:, 2] = 0.
            mesh = meshio.Mesh(points=points, cells=[("triangle", cells.T)])
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plot_mesh(mesh, vals=pred.numpy()[:, 0], with_colorbar=True, levels=levels, border=border)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.subplot(1, 2, 2)
            plot_mesh(mesh, vals=d.y.numpy()[:, 0], with_colorbar=True, levels=levels, border=border)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.show()


if __name__ == "__main__":
    from data2 import get_sdf_data_loader
    from network import GCNet

    n_objects, batch_size, n_epoch = 100, 1, 100
    in_channels, out_channels = 3, 1
    hidden_channels = [16, 64, 128, 64, 16]
    edge_weight = True
    step_size, gamma = 60, 0.5
    lr_0 = 0.001

    loss_func = nn.L1Loss()

    data_folder = "data2/dataset_1/graph1/"
    train_data = get_sdf_data_loader(n_objects, data_folder, batch_size, edge_weight=edge_weight)
    model = GCNet(in_channels, hidden_channels, out_channels)
    train_model(model, train_data, lr_0=lr_0, n_epoch=n_epoch, loss_func=loss_func, step_size=step_size, gamma=gamma)
    plot_results(model, train_data, plot_every=5)
