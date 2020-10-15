import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import GraphUNet, GCNConv, DeepGCNLayer, GENConv
from torch_geometric.utils import (dropout_adj, add_self_loops, sort_edge_index,
                                   remove_self_loops)


class UNet_reg(torch.nn.Module):
    def __init__(self, hidden_layer=256, edge_dropout=False, node_dropout=False):
        super(UNet_reg, self).__init__()
        self.edge_dropout = edge_dropout
        self.node_dropout = node_dropout
        self.unet = GraphUNet(3, hidden_layer, 1, depth=4, pool_ratios=[0.5], sum_res=False)

    def forward(self, data):
        if self.edge_dropout:
            edge_index, _ = dropout_adj(data.edge_index, p=0.8,
                                        force_undirected=True,
                                        num_nodes=data.num_nodes,
                                        training=self.training)
        else:
            edge_index = data.edge_index
        if self.node_dropout:
            x = F.dropout(data.x, p=0.92, training=self.training)
        else:
            x = data.x

        x = self.unet(x, edge_index)

        return x


class GCNet(torch.nn.Module):
    """
    blah
    """
    def __init__(self, in_channels, hidden_channels, out_channels, act=F.relu):
        super(GCNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.act = act

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels[0], improved=True))
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1], improved=True))
        self.convs.append(GCNConv(hidden_channels[-1], out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        """"""
        x = data.x
        edge_index = data.edge_index
        #edge_weight = x.new_ones(edge_index.size(1))
        edge_weight = data.edge_attr

        for i in range(len(self.convs) - 1):
            #edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.act(x)

        #edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
        x = self.convs[-1](x, edge_index, edge_weight)

        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)


class GCNet2(GCNet):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden_lins):
        super(GCNet2, self).__init__(in_channels, hidden_channels, out_channels)
        self.lins = torch.nn.ModuleList()
        self.lins.append(nn.Linear(out_channels, hidden_lins[0]))
        for i in range(len(hidden_lins) - 1):
            self.lins.append(nn.Linear(hidden_lins[i], hidden_lins[i+1]))

        #self.flatten = torch.flatten

    def forward(self, data):
        x = super().forward(data)
        x = x.view(1, -1)
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            x = self.act(x)
        x = self.lins[-1](x)
        return x


class UNet_cls(torch.nn.Module):
    def __init__(self, hidden_layer=256, edge_dropout=False, node_dropout=False):
        super(UNet_cls, self).__init__()
        self.edge_dropout = edge_dropout
        self.node_dropout = node_dropout
        self.unet = GraphUNet(3, hidden_layer, 1, depth=4, pool_ratios=[0.5], sum_res=False)
        self.sig = nn.Sigmoid()

    def forward(self, data):
        if self.edge_dropout:
            edge_index, _ = dropout_adj(data.edge_index, p=0.8,
                                        force_undirected=True,
                                        num_nodes=data.num_nodes,
                                        training=self.training)
        else:
            edge_index = data.edge_index
        if self.node_dropout:
            x = F.dropout(data.x, p=0.92, training=self.training)
        else:
            x = data.x

        x = self.unet(x, edge_index)
        x = self.sig(x)

        return x


class DeeperGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, edge_in_channels):
        super(DeeperGCN, self).__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(edge_in_channels, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, dat):
        x, edge_index, edge_attr = dat.x, dat.edge_index, dat.edge_attr.view(-1, 1)
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        #x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)
