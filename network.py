import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import GraphUNet, GCNConv, GATConv, EdgeConv, FeaStConv
from torch_geometric.utils import (dropout_adj, add_self_loops, sort_edge_index,
                                   remove_self_loops)





class UNet_general(nn.Module):
    def __init__(self, estimator, params, requires_edge_weight=True, **kwargs):
        super().__init__()
        self.act = F.relu
        self.estimator = estimator
        self.params = params
        self.kwargs = kwargs
        self.requires_edge_weight = requires_edge_weight
        self.estimators = torch.nn.ModuleList()
        self.build_estimators()
        self.reset_parameters()

    def build_estimators(self):
        in_channels, hidden_channels, out_channels = self.params
        estimator = self.estimator
        n_channels = len(hidden_channels)
        self.estimators.append(estimator(in_channels, hidden_channels[0]))
        for i in range(n_channels - 1):
            self.estimators.append(estimator(hidden_channels[i], hidden_channels[i], **self.kwargs))
            self.estimators.append(estimator(hidden_channels[i], hidden_channels[i + 1], **self.kwargs))

        for i in range(n_channels, 1, -1):
            self.estimators.append(estimator(hidden_channels[i - 1], hidden_channels[i - 1], **self.kwargs))
            self.estimators.append(estimator(hidden_channels[i - 1] + hidden_channels[i - 2], hidden_channels[i - 2], **self.kwargs))

        self.estimators.append(estimator(hidden_channels[0], hidden_channels[0], **self.kwargs))
        self.estimators.append(estimator(hidden_channels[0], out_channels, **self.kwargs))

    def prep_estimator_input(self, data, x=None, x_concat=None):
        if x is None:
            x = data.x
        edge_idx = data.edge_index
        if x_concat is not None:
            x = torch.cat([x, x_concat], dim=-1)
        if self.requires_edge_weight:
            edge_weight = data.edge_attr
            edge_weight = edge_weight / edge_weight.max()
            return x, edge_idx, edge_weight
        else:
            return x, edge_idx

    def reset_parameters(self):
        for estimator in self.estimators:
            estimator.reset_parameters()

    def forward(self, data):
        xvec = []
        n_channels = len(self.params[1])
        x = None
        for i in range(n_channels):
            inputs = self.prep_estimator_input(data, x=x)
            x = self.estimators[2 * i](*inputs)
            x = self.act(x)
            inputs = self.prep_estimator_input(data, x=x)
            x = self.estimators[2 * i + 1](*inputs)
            x = self.act(x)
            xvec.append(x)

        xvec.pop()
        for i in range(n_channels, 2 * n_channels - 1):
            y = xvec.pop()
            inputs = self.prep_estimator_input(data, x=x, x_concat=y)
            x = self.estimators[2 * i](*inputs)
            x = self.act(x)
            inputs = self.prep_estimator_input(data, x=x)
            x = self.estimators[2 * i + 1](*inputs)
            x = self.act(x)

        inputs = self.prep_estimator_input(data, x=x)
        x = self.estimators[-1](*inputs)
        return x


class UNet_gcn_conv(UNet_general):
    def __init__(self, in_channels, hidden_channels, out_channels):
        params = [in_channels, hidden_channels, out_channels]
        estimator = GCNConv
        super().__init__(estimator, params, requires_edge_weight=True)


class UNet_gat_conv(UNet_general):
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        params = [in_channels, hidden_channels, out_channels]
        estimator = GATConv
        super().__init__(estimator, params, requires_edge_weight=False, **kwargs)


class UNet_edge_conv(UNet_general):
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        params = [in_channels, hidden_channels, out_channels]
        estimator = EdgeConv
        super().__init__(estimator, params, requires_edge_weight=False, **kwargs)


class UNet_feast_conv(UNet_general):
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        params = [in_channels, hidden_channels, out_channels]
        estimator = FeaStConv
        super().__init__(estimator, params, requires_edge_weight=False, **kwargs)


class UNet_spline_conv(UNet_general):
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        params = [in_channels, hidden_channels, out_channels]
        estimator = FeaStConv
        super().__init__(estimator, params, requires_edge_weight=True, **kwargs)


class GCNet(torch.nn.Module):
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
        edge_weight = data.edge_attr
        edge_weight = edge_weight / edge_weight.max()

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.act(x)

        x = self.convs[-1](x, edge_index, edge_weight)

        return x

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)



