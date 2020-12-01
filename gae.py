import numpy as np
import torch_sparse

from sim_pu import prob_labels

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn
from torch_geometric.nn.inits import glorot, zeros
from torch_sparse import matmul, masked_select_nnz
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from sklearn.metrics import roc_auc_score, average_precision_score

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
    def __init__(
        self, in_channels, out_channels, num_dis_nodes, num_comp_nodes, num_pathways
    ):
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
        self.dis_emb = nn.parameter.Parameter(
            torch.rand((num_dis_nodes, out_channels)), requires_grad=True
        )
        self.comp_emb = nn.parameter.Parameter(
            torch.rand((num_comp_nodes, out_channels)), requires_grad=True
        )
        self.path_emb = nn.parameter.Parameter(
            torch.rand((num_pathways, out_channels)), requires_grad=True
        )
        self.lin1 = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU())
        self.normg1 = nn.BatchNorm1d(in_channels // 2)
        self.normg2 = nn.BatchNorm1d(in_channels // 2)
        self.normd1 = nn.BatchNorm1d(in_channels // 2)
        self.normd2 = nn.BatchNorm1d(in_channels // 2)
        self.normc1 = nn.BatchNorm1d(in_channels // 2)
        self.normc2 = nn.BatchNorm1d(in_channels // 2)
        self.normp1 = nn.BatchNorm1d(in_channels // 2)
        self.normp2 = nn.BatchNorm1d(in_channels // 2)

        self.normg3 = nn.BatchNorm1d(out_channels)
        self.normd3 = nn.BatchNorm1d(out_channels)
        self.normc3 = nn.BatchNorm1d(out_channels)
        self.normp3 = nn.BatchNorm1d(out_channels)

        self.conv1 = WRGCNConv(out_channels, in_channels // 2, 3)
        self.conv2 = WRGCNConv(in_channels // 2, in_channels // 2, 3)
        self.conv3 = WRGCNConv(in_channels // 2, out_channels, 3)
        self.drop = nn.Dropout(0.3)

    def forward(self, x, adj_t, edge_types):
        """Calculates embeddings"""
        adj_t = gcn_norm(adj_t, num_nodes=x.size(-2), add_self_loops=False)
        x1 = F.relu(
            self.conv1(
                torch.cat(
                    (self.lin1(x), self.dis_emb, self.comp_emb, self.path_emb), 0
                ),
                adj_t,
                edge_types,
            )
        )
        x1 = torch.cat(
            (
                self.normg1(x1[: x.shape[-2]]),
                self.normd1(x1[x.shape[-2] : x.shape[-2] + self.dis_emb.shape[-2]]),
                self.normc1(
                    x1[
                        x.shape[-2]
                        + self.dis_emb.shape[-2] : x.shape[-2]
                        + self.dis_emb.shape[-2]
                        + self.comp_emb.shape[-2]
                    ]
                ),
                self.normp1(
                    x1[x.shape[-2] + self.dis_emb.shape[-2] + self.comp_emb.shape[-2] :]
                ),
            ),
            0,
        )
        x2 = F.relu(self.conv2(x1, adj_t, edge_types))
        x2 = torch.cat(
            (
                self.normg2(x2[: x.shape[-2]]),
                self.normd2(x2[x.shape[-2] : x.shape[-2] + self.dis_emb.shape[-2]]),
                self.normc2(
                    x2[
                        x.shape[-2]
                        + self.dis_emb.shape[-2] : x.shape[-2]
                        + self.dis_emb.shape[-2]
                        + self.comp_emb.shape[-2]
                    ]
                ),
                self.normp2(
                    x2[x.shape[-2] + self.dis_emb.shape[-2] + self.comp_emb.shape[-2] :]
                ),
            ),
            0,
        )
        x3 = self.conv3(x2, adj_t, edge_types)
        x3 = torch.cat(
            (
                self.normg3(x3[: x.shape[-2]]),
                self.normd3(x3[x.shape[-2] : x.shape[-2] + self.dis_emb.shape[-2]]),
                self.normc3(
                    x3[
                        x.shape[-2]
                        + self.dis_emb.shape[-2] : x.shape[-2]
                        + self.dis_emb.shape[-2]
                        + self.comp_emb.shape[-2]
                    ]
                ),
                self.normp3(
                    x3[x.shape[-2] + self.dis_emb.shape[-2] + self.comp_emb.shape[-2] :]
                ),
            ),
            0,
        )
        x3 = self.drop(x3)
        return x3


class RelDecoder(nn.Module):
    def __init__(self, in_channels, rel_types):
        """Initializes the decoder with encoded embeddings (DistMult)

        sigma(Z @ Ri @ Z.T)
        Ri = diag(rel[i])

        Parameters
        ----------
        in_channels : int
            The number of channels in the node embeddings
        rel_types : int
            The number of relation types
        """
        super(RelDecoder, self).__init__()
        self.in_channels = in_channels
        self.rel_types = rel_types
        self.rel = nn.parameter.Parameter(
            torch.ones((rel_types, in_channels)), requires_grad=True
        )

    def forward(self, z, edge_index, rel_id, sigmoid=True):
        res = ((z[edge_index[0]] * self.rel[rel_id]) * z[edge_index[1]]).sum(dim=1)
        if not sigmoid:
            return res
        return torch.sigmoid(res)


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


def test(z, decoder, entity_types, pos_edge_index, neg_edge_index):
    r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            entity_types (tuple): Ids of the entities.
                against.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = decoder(z, pos_edge_index, entity_types, sigmoid=True)
    neg_pred = decoder(z, neg_edge_index, entity_types, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(y, pred), average_precision_score(y, pred)


def drop_edges(mat, p=0.3):
    mask = torch.rand((mat.storage.row().shape[0],)) > p
    matr = torch_sparse.SparseTensor(
        row=mat.storage.row()[mask],
        col=mat.storage.col()[mask],
        value=torch.ones(mask.sum(), dtype=torch.float),
        sparse_sizes=mat.storage.sparse_sizes(),
    )
    return matr, mask


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
            unique_edge_types = graph.train_edge_types.unique()
            train_pos_adj, dropmask = drop_edges(graph.train_adj_t)
            train_pos_adj = train_pos_adj.to(device)
            edge_types = graph.train_edge_types[dropmask].to(device)
            x = graph.feats.to(device)

            model.train()
            optimizer.zero_grad()
            z = model.encode(x, train_pos_adj, edge_types)

            loss = 0
            for edge_type in unique_edge_types:
                pos_edges = torch.stack(
                    (
                        graph.train_adj_t.storage.row()[~dropmask][
                            graph.train_edge_types[~dropmask] == edge_type
                        ],
                        graph.train_adj_t.storage.col()[~dropmask][
                            graph.train_edge_types[~dropmask] == edge_type
                        ],
                    )
                )
                if pos_edges.shape[-1] != 0:
                    pos_loss = -torch.log(
                        model.decoder(z, pos_edges.to(device), edge_type) + 1e-15
                    ).mean()
                    neg_edges = pos_edges
                    neg_edges[1] = torch.randint(graph.num_nodes, (len(pos_edges[0]),))
                    neg_loss = (
                        -0.5
                        * torch.log(
                            1
                            - model.decoder(z, neg_edges.to(device), edge_type)
                            + 1e-15
                        ).mean()
                        * 0.5
                    )
                    neg_edges = pos_edges
                    neg_edges[0] = torch.randint(graph.num_nodes, (len(pos_edges[0]),))
                    neg_loss = (
                        -0.5
                        * torch.log(
                            1
                            - model.decoder(z, neg_edges.to(device), edge_type)
                            + 1e-15
                        ).mean()
                    )
                    loss += graph.loss_weights[edge_type] * (
                        graph.pos_multiplier * pos_loss + neg_loss
                    )

            loss.backward()
            optimizer.step()
            losses.append(loss.item() / np.sum(graph.loss_weights))

            model.eval()
            with torch.no_grad():
                z = model.encode(x, train_pos_adj, edge_types)
                auc, ap = test(
                    z,
                    model.decoder,
                    1,
                    graph.pos_val[1].to(device),
                    graph.neg_val[1].to(device),
                )
                aucs.append(auc)
                aps.append(ap)

        mean_auc = np.sum(aucs[-len(loader) :]) / len(loader)
        mean_ap = np.sum(aps[-len(loader) :]) / len(loader)
        mean_loss = np.sum(losses[-len(loader) :]) / len(loader)
        if scheduler is not None:
            scheduler.step()

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
