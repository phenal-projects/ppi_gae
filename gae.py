import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn


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
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        return torch.cat([self.conv3(x2, edge_index), x2, x1], -1)


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
                auc, _ = model.test(
                    z,
                    graph.val_pos_edge_index.to(device),
                    graph.val_neg_edge_index.to(device),
                )
            aucs.append(auc)

        mean_auc = sum(aucs[-len(loader) :]) / len(loader)
        if scheduler is not None:
            scheduler.step(-mean_auc)

        if callback is not None:
            if callback(model, mean_auc):
                return model
    return model


def finetune_gae(
    model, loader, loss_fn, optimizer, scheduler, device, epochs, callback=None
):
    """Finetunes Graph Autoencoder

    NEEDS REFACTORING! DOES NOT WORK YET!

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
    raise NotImplementedError
    model.to(device)
    losses = []
    val_losses = []

    for epoch in range(epochs):
        for graph in loader:

            train_pos_adj = graph.adj_t.to(device)
            x = graph.x.to(device)
            train_mask = torch.logical_and(
                graph.train_nodes_mask, graph.expr_mask
            )  # may cause problems when many batches

            model.train()
            optimizer.zero_grad()
            z = model(x, train_pos_adj)
            loss = loss_fn(z[train_mask], graph.y[train_mask].to(device),)
            val_loss = loss_fn(
                z[~graph.train_nodes_mask],
                graph.y[~graph.train_nodes_mask].to(device),
            )
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
