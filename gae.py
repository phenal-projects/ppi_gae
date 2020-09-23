import numpy as np

from sim_pu import prob_labels

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

# deterministic behaviour
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)


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
        self.conv2 = gnn.GCNConv(
            2 * out_channels, 2 * out_channels, cached=False
        )
        self.conv3 = gnn.GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        """Calculates embeddings"""
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(F.relu(x1), edge_index)
        return torch.cat([self.conv3(F.relu(x2), edge_index), x2, x1], -1)


class CTDEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, drug_nodes):
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
        self.out_channels = out_channels * 5
        self.emb = nn.parameter.Parameter(
            torch.rand((drug_nodes, in_channels)), requires_grad=True
        )
        self.conv1_gg = gnn.GCNConv(
            in_channels, 2 * out_channels, cached=False
        )
        self.conv2_gg = gnn.GCNConv(
            2 * out_channels, 4 * out_channels, cached=False
        )
        self.conv3_gg = gnn.GCNConv(
            4 * out_channels, out_channels, cached=False
        )
        self.conv1_gd = gnn.GCNConv(
            in_channels, 2 * out_channels, cached=False
        )
        self.conv2_gd = gnn.GCNConv(
            2 * out_channels, 4 * out_channels, cached=False
        )
        self.conv3_gd = gnn.GCNConv(
            4 * out_channels, out_channels, cached=False
        )

    def forward(self, x, adj_t_gg, adj_t_gd):
        """Calculates embeddings"""
        x1 = self.conv1_gg(torch.cat((x, self.emb)), adj_t_gg) + self.conv1_gd(
            torch.cat((x, self.emb)), adj_t_gd
        )
        x2 = self.conv2_gg(F.relu(x1), adj_t_gg) + self.conv2_gd(
            F.relu(x1), adj_t_gd
        )
        x3 = self.conv3_gg(F.relu(x2), adj_t_gg) + self.conv3_gd(
            F.relu(x2), adj_t_gd
        )
        return torch.cat((x1, x2, x3), -1)


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


def train_gae(
    model, loader, optimizer, scheduler, device, epochs, callback=None
):
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
                prob_labels(graph.y[train_mask], graph.probs[train_mask]).to(
                    device
                ),
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


def train_ctd_gae(
    model, loader, optimizer, scheduler, device, epochs, callback=None
):
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
    aucs_gd = []
    aps_gd = []
    aucs_gg = []
    aps_gg = []

    for epoch in range(epochs):
        for graph in loader:

            train_pos_adj_gg = graph.train_adj_t_gg.to(device)
            train_pos_adj_gd = graph.train_adj_t_gd.to(device)
            x = graph.feats.to(device)

            model.train()
            optimizer.zero_grad()
            graph = graph
            z = model.encode(x, train_pos_adj_gg, train_pos_adj_gd)
            loss = (
                graph.pos_train_gg.shape[1]
                / (graph.pos_train_gd.shape[1] + graph.pos_train_gg.shape[1])
                * F.binary_cross_entropy(
                    model.decode(z, graph.pos_train_gd.to(device)),
                    torch.ones(
                        graph.pos_train_gd.shape[1],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
            )
            loss += (
                graph.pos_train_gg.shape[1]
                / (graph.pos_train_gd.shape[1] + graph.pos_train_gg.shape[1])
                * F.binary_cross_entropy(
                    model.decode(z, graph.neg_train_gd.to(device)),
                    torch.zeros(
                        graph.neg_train_gd.shape[1],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
            )
            loss += (
                graph.pos_train_gd.shape[1]
                / (graph.pos_train_gd.shape[1] + graph.pos_train_gg.shape[1])
                * F.binary_cross_entropy(
                    model.decode(z, graph.pos_train_gg.to(device)),
                    torch.ones(
                        graph.pos_train_gg.shape[1],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
            )
            loss += (
                graph.pos_train_gd.shape[1]
                / (graph.pos_train_gd.shape[1] + graph.pos_train_gg.shape[1])
                * F.binary_cross_entropy(
                    model.decode(z, graph.neg_train_gg.to(device)),
                    torch.zeros(
                        graph.neg_train_gg.shape[1],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item() / 2)

            model.eval()
            with torch.no_grad():
                z = model.encode(x, train_pos_adj_gg, train_pos_adj_gd)
                auc, ap = model.test(
                    z,
                    graph.pos_val_gd.to(device),
                    graph.neg_val_gd.to(device),
                )
                aucs_gd.append(auc)
                aps_gd.append(ap)
                auc, ap = model.test(
                    z,
                    graph.pos_val_gg.to(device),
                    graph.neg_val_gg.to(device),
                )
                aucs_gg.append(auc)
                aps_gg.append(ap)

        mean_auc_gd = np.sum(aucs_gd[-len(loader) :]) / len(loader)
        mean_ap_gd = np.sum(aps_gd[-len(loader) :]) / len(loader)
        mean_auc_gg = np.sum(aucs_gg[-len(loader) :]) / len(loader)
        mean_ap_gg = np.sum(aps_gg[-len(loader) :]) / len(loader)
        mean_loss = np.sum(losses[-len(loader) :]) / len(loader)
        if scheduler is not None:
            scheduler.step(mean_loss)

        if callback is not None:
            if callback(
                model,
                (mean_auc_gd, mean_ap_gd, mean_auc_gg, mean_ap_gg, mean_loss),
            ):
                return model
    return model


def encode_ctd(model, graph, device):
    """Encodes graph nodes with the given model"""
    model = model.to(device)
    with torch.no_grad():
        z = model.encode(
            graph.feats.to(device),
            graph.adj_t_gg.to(device),
            graph.adj_t_gd.to(device),
        )
    return z.detach().cpu().numpy()
