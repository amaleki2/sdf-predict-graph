class UNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, with_middle_output=False):
        super().__init__()
        self.act = F.relu
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_middle_output = with_middle_output
        self.convs = torch.nn.ModuleList()
        self.n_channels = len(hidden_channels)
        self.convs.append(GCNConv(in_channels, hidden_channels[0], improved=True))
        for i in range(self.n_channels - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i + 1], improved=True))
        for i in range(1, self.n_channels):
            self.convs.append(GCNConv(hidden_channels[self.n_channels - i] + hidden_channels[self.n_channels - i - 1],
                                      hidden_channels[self.n_channels - i - 1], improved=True))
        self.convs.append(GCNConv(hidden_channels[0], out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        """"""
        xvec = []
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr

        for i in range(self.n_channels):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.act(x)
            xvec.append(x)
        z = xvec.pop()
        for i in range(self.n_channels, 2 * self.n_channels - 1):
            y = xvec.pop()
            x = torch.cat([x, y], dim=-1)
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.act(x)

        x = self.convs[-1](x, edge_index, edge_weight)
        if self.with_middle_output:
            return x, z
        else:
            return x


class UNet2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, with_middle_output=False):
        super().__init__()
        self.act = F.relu
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_middle_output = with_middle_output
        self.convs = torch.nn.ModuleList()
        self.n_channels = len(hidden_channels)
        self.convs.append(GCNConv(in_channels, hidden_channels[0], improved=True))
        for i in range(self.n_channels - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i], improved=True))
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i + 1], improved=True))

        for i in range(self.n_channels, 1, -1):
            self.convs.append(GCNConv(hidden_channels[i - 1],  hidden_channels[i - 1], improved=True))
            self.convs.append(GCNConv(hidden_channels[i - 1] + hidden_channels[i - 2], hidden_channels[i - 2], improved=True))

        self.convs.append(GCNConv(hidden_channels[0], hidden_channels[0], improved=True))
        self.convs.append(GCNConv(hidden_channels[0], out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        """"""
        xvec = []
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr

        for i in range(self.n_channels):
            x = self.convs[2 * i](x, edge_index, edge_weight)
            x = self.act(x)
            x = self.convs[2 * i + 1](x, edge_index, edge_weight)
            x = self.act(x)
            xvec.append(x)
        z = xvec.pop()
        for i in range(self.n_channels, 2 * self.n_channels - 1):
            y = xvec.pop()
            x = torch.cat([x, y], dim=-1)
            x = self.convs[2 * i](x, edge_index, edge_weight)
            x = self.act(x)
            x = self.convs[2 * i + 1](x, edge_index, edge_weight)
            x = self.act(x)

        x = self.convs[-1](x, edge_index, edge_weight)
        if self.with_middle_output:
            return x, z
        else:
            return x
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


class GCNet3(torch.nn.Module):
    """
    blah
    """
    def __init__(self, in_channels, hidden_channels, out_channels, act=F.relu):
        super(GCNet3, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.act = act

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels[0], improved=True))
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GCNConv(hidden_channels[i]+in_channels, hidden_channels[i+1], improved=True))
        self.convs.append(GCNConv(hidden_channels[-1]+in_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):

        """"""
        x0 = data.x
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr

        for i in range(len(self.convs) - 1):
            if i > 0:
                x = torch.cat((x, x0), dim=-1)
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.act(x)

        # edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
        x = torch.cat((x, x0), dim=-1)
        x = self.convs[-1](x, edge_index, edge_weight)

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


class UNet3(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, estimator=GCNConv, with_middle_output=False):
        super().__init__()
        self.act = F.relu
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_middle_output = with_middle_output
        self.convs = torch.nn.ModuleList()
        self.n_channels = len(hidden_channels)
        self.convs.append(estimator(in_channels, hidden_channels[0]))
        for i in range(self.n_channels - 1):
            self.convs.append(estimator(hidden_channels[i], hidden_channels[i]))
            self.convs.append(estimator(hidden_channels[i], hidden_channels[i + 1]))

        for i in range(self.n_channels, 1, -1):
            self.convs.append(estimator(hidden_channels[i - 1], hidden_channels[i - 1]))
            self.convs.append(estimator(hidden_channels[i - 1] + hidden_channels[i - 2], hidden_channels[i - 2]))

        self.convs.append(estimator(hidden_channels[0], hidden_channels[0]))
        self.convs.append(estimator(hidden_channels[0], out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        """"""
        xvec = []
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr
        edge_weight = edge_weight / edge_weight.max()

        for i in range(self.n_channels):
            x = self.convs[2 * i](x, edge_index, edge_weight)
            x = self.act(x)
            x = self.convs[2 * i + 1](x, edge_index, edge_weight)
            x = self.act(x)
            xvec.append(x)
        z = xvec.pop()
        for i in range(self.n_channels, 2 * self.n_channels - 1):
            y = xvec.pop()
            x = torch.cat([x, y], dim=-1)
            x = self.convs[2 * i](x, edge_index, edge_weight)
            x = self.act(x)
            x = self.convs[2 * i + 1](x, edge_index, edge_weight)
            x = self.act(x)

        x = self.convs[-1](x, edge_index, edge_weight)
        if self.with_middle_output:
            return x, z
        else:
            return x


class UNet4(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, with_middle_output=False):
        super().__init__()
        self.act = F.relu
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_middle_output = with_middle_output
        self.convs = torch.nn.ModuleList()
        self.n_channels = len(hidden_channels)
        self.convs.append(GCNConv(in_channels, hidden_channels[0]))
        for i in range(self.n_channels - 1):
            self.convs.append(GATConv(hidden_channels[i], hidden_channels[i]))
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i + 1]))

        for i in range(self.n_channels, 1, -1):
            self.convs.append(GATConv(hidden_channels[i - 1], hidden_channels[i - 1]))
            self.convs.append(GATConv(hidden_channels[i - 1] + hidden_channels[i - 2], hidden_channels[i - 2]))

        self.convs.append(GATConv(hidden_channels[0], hidden_channels[0]))
        self.convs.append(GATConv(hidden_channels[0], out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        """"""
        xvec = []
        x = data.x
        edge_index = data.edge_index

        for i in range(self.n_channels):
            x = self.convs[2 * i](x, edge_index)
            x = self.act(x)
            x = self.convs[2 * i + 1](x, edge_index)
            x = self.act(x)
            xvec.append(x)
        z = xvec.pop()
        for i in range(self.n_channels, 2 * self.n_channels - 1):
            y = xvec.pop()
            x = torch.cat([x, y], dim=-1)
            x = self.convs[2 * i](x, edge_index)
            x = self.act(x)
            x = self.convs[2 * i + 1](x, edge_index)
            x = self.act(x)

        x = self.convs[-1](x, edge_index)
        if self.with_middle_output:
            return x, z
        else:
            return x
