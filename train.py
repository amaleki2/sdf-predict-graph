import os
import sys
import meshio
import numpy as np
import matplotlib.pyplot as plt
from data2 import plot_mesh, plot_mesh_onto_line

from scipy.spatial import Delaunay

import torch
import torch.nn as nn

l1_loss = nn.L1Loss
l2_loss = nn.MSELoss


def borderless_loss(pred, target, loss_func, data, radius):
    mask = torch.logical_and(torch.abs(data.x[:, 0]) < 1 - radius,
                             torch.abs(data.x[:, 1]) < 1 - radius)
    loss = loss_func(reduction='none')(pred, target)
    loss_masked = loss[mask]
    loss_masked_reduced = torch.mean(loss_masked)
    mask2 = torch.abs(data.y < 0)
    loss_inner = loss[mask2]
    loss_inner_reduced = torch.mean(loss_inner)
    loss_final = loss_masked_reduced + loss_inner_reduced * 10
    return loss_final


def eikonal_loss(pred, xy, device='cuda', retain_graph=True):
    pred.backward(gradient=torch.ones(pred.size()).to(device), retain_graph=retain_graph)
    dg = xy.grad[:, :2]
    dg_mag = torch.sqrt(torch.sum(dg * dg, dim=-1))
    eikonal_loss = l2_loss()(dg_mag, torch.ones(dg_mag.size()).to(device))
    eikonal_loss.requires_grad = True
    return eikonal_loss


def find_best_gpu():
    # this function finds the GPU with most free memory.
    if 'linux' in sys.platform and torch.cuda.device_count() > 1:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu_id = np.argmax(memory_available).item()
        print("best gpu is %d with %0.1f Gb available space" %(gpu_id, memory_available[gpu_id]/1000))
        return gpu_id


def train_model(model, train_data, lr_0=0.001, n_epoch=101, loss_func=l1_loss,
                with_borderless_loss=True, with_eikonal_loss=False,
                print_every=10, step_size=50, gamma=0.5, radius=0.1, save_name=""):
    print("training begins")
    device = torch.device('cuda')
    gpu_id = find_best_gpu()
    if gpu_id:
        torch.cuda.set_device(gpu_id)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_0)  # , weight_decay=0.001)
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
                loss = loss_func()(pred, target)

            loss += eikonal_loss(pred, d.x) if with_eikonal_loss else 0
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        running_loss_list.append(running_loss / len(train_data))
        if epoch % print_every == 0:
            print("epoch=%d, loss=%0.5e, lr=%0.5e" % (epoch, running_loss / len(train_data),
                                                      optimizer.param_groups[0]['lr']))
            torch.save(model.state_dict(), "models/model" + save_name + ".pth")
            np.save("models/loss" + save_name + ".npy", running_loss_list)


def plot_results(model, data, ndata=5, levels=None, border=None, save_name=""):
    loss_history = np.load("models/loss" + save_name + ".npy")
    plt.plot(loss_history)
    plt.yscale('log')

    device = 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load("models/model" + save_name + ".pth", map_location=device))
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(data):
            if i > ndata:
                break
            d = d.to(device=device)
            pred = model(d)
            cells = d.face.numpy()
            points = d.x.numpy()
            points[:, 2] = 0.
            mesh = meshio.Mesh(points=points, cells=[("triangle", cells.T)])

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plot_mesh(mesh, vals=pred.numpy()[:, 0], with_colorbar=False, levels=levels, border=border)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.subplot(1, 2, 2)
            p = plot_mesh(mesh, vals=d.y.numpy()[:, 0], with_colorbar=False, levels=levels, border=border)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.gcf().subplots_adjust(right=0.8)
            cbar_ax = plt.gcf().add_axes([0.85, 0.15, 0.05, 0.7])
            plt.gcf().colorbar(p, cax=cbar_ax)
            plt.show()

            
def plot_results_diff(model, data, ndata=5, levels=None, border=None, save_name=""):
    device = 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load("models/model" + save_name + ".pth", map_location=device))
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(data):
            if i > ndata:
                break
            d = d.to(device=device)
            pred = model(d)
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
            p = plot_mesh(mesh, vals=pred.numpy()[:, 0] - d.y.numpy()[:, 0], with_colorbar=True, levels=None, border=border)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.show()


def plot_results_pxl(model, data, ndata=5, save_name="", levels=[-0.2, 0, 0.4, .8]):
    loss_history = np.load("models/loss" + save_name + ".npy")
    plt.plot(loss_history)
    plt.yscale('log')

    device = 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load("models/model" + save_name + ".pth", map_location=device))
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(data):            
            if i > ndata:
                break
            d = d.to(device=device)
            pred = model(d)
            points = d.x.numpy()
            points[:, 2] = 0.
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.contourf(pred.reshape(128, 128))
            cn = plt.contour(pred.reshape(128, 128), levels=levels, colors='w')
            plt.clabel(cn, fmt='%0.2f', colors='k', fontsize=10)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.subplot(1, 2, 2)
            p = plt.contourf(d.y.numpy().reshape(128, 128))
            cn = plt.contour(d.y.numpy().reshape(128, 128), levels=levels, colors='w')
            plt.clabel(cn, fmt='%0.2f', colors='k', fontsize=10)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.gcf().subplots_adjust(right=0.8)
            cbar_ax = plt.gcf().add_axes([0.85, 0.15, 0.05, 0.7])
            plt.gcf().colorbar(p, cax=cbar_ax)
            plt.show()

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            for idx in [30, 64, 100]:
                plt.plot(np.linspace(-1, 1, 128), pred.reshape(128, 128)[idx, :])
                plt.plot(np.linspace(-1, 1, 128), d.y.numpy().reshape(128, 128)[idx, :], linestyle='--')
            plt.subplot(1, 2, 2)
            for idx in [30, 64, 100]:
                plt.plot(np.linspace(-1, 1, 128), pred.reshape(128, 128)[:, idx])
                plt.plot(np.linspace(-1, 1, 128), d.y.numpy().reshape(128, 128)[:, idx], linestyle='--')

                
def plot_results_over_line(model, data, lines=(-0.5, 0, 0.5), ndata=5, save_name=""):
    device = 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load("models/model" + save_name + ".pth", map_location=device))
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(data):
            if i > ndata:
                break
            d = d.to(device=device)
            pred = model(d)
            pred = pred.numpy()[:, 0]
            gt = d.y.numpy()[:, 0]

            cells = d.face.numpy()
            points = d.x.numpy()
            points[:, 2] = 0.
            mesh = meshio.Mesh(points=points, cells=[("triangle", cells.T)])

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            for line in lines:
                plot_mesh_onto_line(mesh, val=pred, x=line)
                plot_mesh_onto_line(mesh, val=gt, x=line, linestyle="--")

            plt.subplot(1, 2, 2)
            for line in lines:
                plot_mesh_onto_line(mesh, val=pred, y=line)
                plot_mesh_onto_line(mesh, val=gt, y=line, linestyle="--")
            plt.show()
            
            
def plot_results_for_cells(model, data, ndata=5, levels=None, border=None, save_name=""):
    loss_history = np.load("models/loss" + save_name + ".npy")
    plt.plot(loss_history)
    plt.yscale('log')

    device = 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load("models/model" + save_name + ".pth", map_location=device))
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(data):
            if i > ndata:
                break
            d = d.to(device=device)
            pred = model(d)
            xmean = np.mean(d.x.numpy()[:, 0::3], axis=-1)
            ymean = np.mean(d.x.numpy()[:, 1::3], axis=-1)
            points = np.stack((xmean, ymean))
            tri = Delaunay(points.T)
            cells = tri.simplices
            zmean = np.zeros_like(xmean)
            points = np.stack((xmean, ymean, zmean))
            mesh = meshio.Mesh(points=points.T, cells=[("triangle", cells)])

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plot_mesh(mesh, vals=pred.numpy()[:, 0], with_colorbar=False, levels=levels, border=border)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.subplot(1, 2, 2)
            p = plot_mesh(mesh, vals=d.y.numpy()[:, 0], with_colorbar=False, levels=levels, border=border)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.gcf().subplots_adjust(right=0.8)
            cbar_ax = plt.gcf().add_axes([0.85, 0.15, 0.05, 0.7])
            plt.gcf().colorbar(p, cax=cbar_ax)
            plt.show()