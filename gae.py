import numpy as np

from sim_pu import prob_labels

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn
from torch_geometric.nn.inits import glorot, zeros
from torch_sparse import matmul, masked_select_nnz
from torch_geometric.nn.conv.gcn_conv import gcn_norm

# deterministic behaviour
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)


class WRGCNConv(gnn.MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations):
        """WRGCN layer

        Parameters
        ----------
        in_channels : int
            The number of input dimensions
        out_channels : int
            The number of output dimensions
        num_relations : int
            The number of edge types
        """
        super(WRGCNConv, self).__init__(aggr="add", node_dim=-2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        # parameters
        self.weight = nn.Parameter(
            torch.Tensor(num_relations, in_channels, out_channels)
        )
        self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x, adj, edge_type):
        """Forward-passing

        Parameters
        ----------
        x : torch.Tensor
            The node features
        adj : torch_sparse.SparseMatrix
            Graph's adjacency matrix
        edge_type : torch.LongTensor
            The types of the edges

        Returns
        -------
        torch.Tensor
            The new node features
        """
        # adj normalization. Does not use edge classes!
        out = x @ self.root + self.bias
        for i in range(self.num_relations):
            tmp = masked_select_nnz(adj, edge_type == i, layout="coo")
            h = self.propagate(tmp, x=x, size=(x.size(-2), x.size(-2)))
            out = out + h @ self.weight[i]
        return out

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce="add")

    def __repr__(self):
        return "{}({}, {}, num_relations={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_relations,
        )


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Initializes the GCN encoder

        Parameters
        ----------
        in_channels : int
            The number of channels in the input graph nodes
        out_channels : int
            The number of dimensions in the embeddings
        """
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels * 5
        self.conv1 = gnn.GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv2 = gnn.GCNConv(2 * out_channels, 2 * out_channels, cached=False)
        self.conv3 = gnn.GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        """Calculates embeddings"""
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(F.relu(x1), edge_index)
        return torch.cat([self.conv3(F.relu(x2), edge_index), x2, x1], -1)


class CTDEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, dis_nodes):
        """Initializes the GCN encoder with encoded embeddings

        Parameters
        ----------
        in_channels : int
            The number of channels in the input graph nodes
        out_channels : int
            The number of dimensions in the embeddings
        """
        super(CTDEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb = nn.parameter.Parameter(
            torch.rand((dis_nodes, in_channels)), requires_grad=True
        )
        self.norm1 = nn.BatchNorm1d(2 * out_channels)
        self.norm2 = nn.BatchNorm1d(4 * out_channels)
        self.conv1 = WRGCNConv(in_channels, 2 * out_channels, 3)
        self.conv2 = WRGCNConv(2 * out_channels, 4 * out_channels, 3)
        self.convinf = WRGCNConv(4 * out_channels, 4 * out_channels, 3)
        self.conv3 = WRGCNConv(4 * out_channels, out_channels, 3)

    def forward(self, x, adj_t, edge_types):
        """Calculates embeddings"""
        adj_t = gcn_norm(adj_t, num_nodes=x.size(-2), add_self_loops=False)
        x1 = self.norm1(self.conv1(torch.cat((x, self.emb), 0), adj_t, edge_types))
        x2 = self.norm2(self.conv2(F.relu(x1), adj_t, edge_types))
        for i in range(5):
            x2 = self.norm2(self.convinf(F.relu(x2), adj_t, edge_types))
        x3 = self.conv3(F.relu(x2), adj_t, edge_types)
        return x3


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, nodes):
        """Initializes the GCN encoder with encoded embeddings

        Parameters
        ----------
        in_channels : int
            The number of channels in the input graph nodes
        out_channels : int
            The number of dimensions in the embeddings
        """
        super(SimpleEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels * 6
        self.emb = nn.parameter.Parameter(
            torch.rand((nodes, out_channels)), requires_grad=True
        )

    def forward(self, *args):
        """Calculates embeddings"""
        return self.emb


def train_gae(model, loader, optimizer, scheduler, device, epochs, callback=None):
    """Trains Graph Autoencoder

    Parameters
    ----------
    model : torch_geometric.nn.GAE
        GAE model to train
    loader : any graph loader
        Loader for GAE training. The graphs should contain sparse matrix!
    optimizer : any optimizer
        An optimizer for training
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler
    device : str
        Device string. The device will be used for training
    epochs : int
        A number of epochs to train
    callback : callable, optional
        A function that is called in the end of epochs, if returns true, the
        training stops, by default None

    Returns
    -------
    torch_geometric.nn.GAE
        Trained model
    """
    model.to(device)
    losses = []
    aucs = []
    aps = []
    for epoch in range(epochs):
        for graph in loader:

            train_pos_adj = graph.adj_t.to(device)
            x = graph.x.to(device)

            model.train()
            optimizer.zero_grad()
            graph = graph
            z = model.encode(x, train_pos_adj)
            loss = model.recon_loss(z, graph.train_pos_edge_index.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                z = model.encode(x, train_pos_adj)
                auc, ap = model.test(
                    z,
                    graph.val_pos_edge_index.to(device),
                    graph.val_neg_edge_index.to(device),
                )
            aucs.append(auc)
            aps.append(ap)

        mean_auc = np.sum(aucs[-len(loader) :]) / len(loader)
        mean_ap = np.sum(aps[-len(loader) :]) / len(loader)
        if scheduler is not None:
            scheduler.step(-mean_auc)

        if callback is not None:
            if callback(model, (mean_auc, mean_ap)):
                return model
    return model


def finetune_gae(
    model, loader, loss_fn, optimizer, scheduler, device, epochs, callback=None
):
    """Finetunes Graph Autoencoder

    Parameters
    ----------
    model : torch_geometric.nn.GAE
        GAE Encoder with classification head.
    loader : any graph loader
        Loader for GAE training. The graphs should contain sparse matrix and
        labels!
    loss_fn : callable
        Loss function
    optimizer : any optimizer
        An optimizer for training
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler
    device : str
        Device string. The device will be used for training
    epochs : int
        A number of epochs to train
    callback : callable, optional
        A function that is called in the end of epochs, if returns true, the
        training stops, by default None

    Returns
    -------
    torch_geometric.nn.GAE
        Trained model
    """
    model.to(device)
    losses = []
    val_losses = []

    for epoch in range(epochs):
        for graph in loader:

            train_pos_adj = graph.adj_t.to(device)
            x = graph.x.to(device)

            expr_mask = torch.zeros((len(graph.x),), dtype=torch.bool)
            expr_mask[graph.new_id] = True
            train_mask = torch.logical_and(
                graph.train_nodes_mask, expr_mask
            )  # may cause problems when many batches
            val_mask = torch.logical_and(
                torch.logical_not(graph.train_nodes_mask), expr_mask
            )

            model.train()
            optimizer.zero_grad()
            z = model(x, train_pos_adj)
            loss = loss_fn(
                z[train_mask],
                prob_labels(graph.y[train_mask], graph.probs[train_mask]).to(device),
            )
            val_loss = loss_fn(z[val_mask], graph.y[val_mask].to(device),)
            loss.backward()
            optimizer.step()

            # history
            losses.append(loss.item())
            val_losses.append(val_loss.item())

        mean_loss = sum(val_losses[-len(loader) :]) / len(loader)
        if scheduler is not None:
            scheduler.step(mean_loss)

        if callback is not None:
            if callback(model, mean_loss):
                return model
    return model


def encode(model, graph, device):
    """Encodes graph nodes with the given model"""
    model = model.to(device)
    with torch.no_grad():
        z = model.encode(graph.x.to(device), graph.adj_t.to(device))
    return z.cpu().numpy()


def train_ctd_gae(model, loader, optimizer, scheduler, device, epochs, callback=None):
    """Trains CTD Graph Autoencoder

    Parameters
    ----------
    model : torch_geometric.nn.GAE
        GAE model to train
    loader : any graph loader
        Loader for GAE training. The graphs should contain sparse matrix!
    optimizer : any optimizer
        An optimizer for training
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler
    device : str
        Device string. The device will be used for training
    epochs : int
        A number of epochs to train
    callback : callable, optional
        A function that is called in the end of epochs, if returns true, the
        training stops, by default None

    Returns
    -------
    torch_geometric.nn.GAE
        Trained model
    """
    model.to(device)

    losses = []
    aucs = []
    aps = []

    for epoch in range(epochs):
        for graph in loader:

            train_pos_adj = graph.train_adj_t.to(device)
            edge_types = graph.train_edge_types.to(device)
            x = graph.feats.to(device)

            model.train()
            optimizer.zero_grad()
            graph = graph
            z = model.encode(x, train_pos_adj, edge_types)
            pos_loss = -torch.log(
                model.decoder(z, graph.pos_train_gd.to(device), sigmoid=True) + 1e-15
            ).mean()
            neg_loss = -torch.log(
                1
                - model.decoder(z, graph.neg_train_gd.to(device), sigmoid=True)
                + 1e-15
            ).mean()
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                z = model.encode(x, train_pos_adj, edge_types)
                auc, ap = model.test(
                    z, graph.pos_val_gd.to(device), graph.neg_val_gd.to(device),
                )
                aucs.append(auc)
                aps.append(ap)

        mean_auc = np.sum(aucs[-len(loader) :]) / len(loader)
        mean_ap = np.sum(aps[-len(loader) :]) / len(loader)
        mean_loss = np.sum(losses[-len(loader) :]) / len(loader)
        if scheduler is not None:
            scheduler.step(mean_loss)

        if callback is not None:
            if callback(model, (mean_auc, mean_ap, mean_loss),):
                return model
    return model


def encode_ctd(model, graph, device):
    """Encodes graph nodes with the given model"""
    model = model.to(device)
    with torch.no_grad():
        z = model.encode(
            graph.feats.to(device), graph.adj_t.to(device), graph.edge_types.to(device),
        )
    return z.detach().cpu().numpy()
