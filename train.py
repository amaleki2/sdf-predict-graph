import torch
import torch.nn as nn

from network import GCNet
from data2 import *


def eikonal_loss(pred, xy, device='cuda'):
    pred.backward(gradient=torch.ones(pred.size()).to(device), retain_graph=True)
    dg = xy.grad[:, :2]
    dg_mag = torch.sqrt(torch.sum(dg * dg, dim=-1))
    eikonal_loss = nn.MSELoss()(dg_mag, torch.ones(dg_mag.size()))
    return eikonal_loss


def train_model(model, train_data, lr_0=0.001, n_epoch=101,
                loss_func=nn.L1Loss(), with_eikonal_loss=False,
                print_every=10):
    print("training begins")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_0, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
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
            loss = loss_func(pred, target)
            loss += eikonal_loss(pred, d.x) if with_eikonal_loss else 0
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        if epoch % print_every == 0:
            print("epoch=%d, loss=%0.5e, lr=%0.5e" % (epoch, running_loss / len(train_data),
                                                      optimizer.param_groups[0]['lr']))
        running_loss_list.append(running_loss / len(train_data))

    torch.save(model.state_dict(), "models/model" + ".pth")
    np.save("models/loss.npy", running_loss_list)


def plot_results(model, data, plot_every=-1):
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
            cells = d.face.numpy()
            points = d.x.numpy()
            points[:, 2] = 0.
            mesh = meshio.Mesh(points=points, cells=[("triangle", cells.T)])
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plot_mesh(mesh, vals=pred.numpy()[:, 0])
            plt.subplot(1, 2, 2)
            plot_mesh(mesh, vals=d.y.numpy()[:, 0])
            plt.show()

n_objects, batch_size = 100, 1
data_folder = "data2/dataset_1/graph1/"
in_channels, out_channels = 3, 1
hidden_channels = [16, 64, 128, 64, 16]

model = GCNet(in_channels, hidden_channels, out_channels)
train_data = get_sdf_data_loader(n_objects, data_folder, batch_size)
loss_func = nn.L1Loss()
train_model(model, train_data, lr_0=0.001, n_epoch=101, loss_func=loss_func)
plot_results(model, train_data, plot_every=10)
