"""CTD Gene-Disease GAE training"""
from sys import argv
import argparse

import numpy as np

import torch
import torch.optim as opt
import torch_geometric.nn as gnn
import torch_geometric.data as gdata
from torch_geometric import utils
from torch_geometric.utils.negative_sampling import negative_sampling
from torch_sparse import SparseTensor

import mlflow
import mlflow.pytorch

import gae


BEST_AP = 0
ESTOP_COUNTER = 0
EPOCH_COUNTER = 0


def construct_parser():
    """Constructs args parser"""
    parser = argparse.ArgumentParser(description="Train HGAE.")
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("wd", type=float, help="weight decay")
    parser.add_argument(
        "epochs", type=int, help="the number of epochs to train"
    )
    parser.add_argument(
        "val_year",
        type=int,
        help="after this year the edges will be considered val",
    )
    parser.add_argument(
        "test_year",
        type=int,
        help="after this year the edges will be considered test",
    )
    parser.add_argument("dim", type=int, help="the size of embeddings")
    parser.add_argument("device", type=str, help="cuda or cpu")
    parser.add_argument("seed", type=int, help="random seed for repruduction")

    # Paths
    parser.add_argument(
        "edges", type=str, help="a path to the edge list (npy)"
    )
    parser.add_argument(
        "edges_dates",
        type=str,
        help="a list of years for train/val/test split",
    )
    parser.add_argument(
        "features",
        type=str,
        help="a path to the features of the protein nodes (npy)",
    )
    parser.add_argument(
        "node_classes",
        type=str,
        help="a path to the expressions data file (npy)",
    )
    return parser


def callback(model, auc_ap_loss):
    """A callback function for printing training stats"""
    global BEST_AP
    global ESTOP_COUNTER
    global EPOCH_COUNTER
    auc_gd, ap_gd, loss = auc_ap_loss
    if BEST_AP < ap_gd:
        BEST_AP = ap_gd
        ESTOP_COUNTER = 0
        torch.save(model, "./best_model.pt")
    ESTOP_COUNTER += 1
    mlflow.log_metric("ROC_AUC_GD", auc_gd, step=EPOCH_COUNTER)
    mlflow.log_metric("AP_GD", ap_gd, step=EPOCH_COUNTER)
    mlflow.log_metric("LOSS", loss, step=EPOCH_COUNTER)
    EPOCH_COUNTER += 1
    if ESTOP_COUNTER > 500:
        print("Stop!")
        return True
    return False


args = construct_parser().parse_args(argv[1:])

# deterministic behaviour
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# load the data
edge_dates = torch.LongTensor(np.load(args.edges_dates))
edge_index, edge_dates = utils.remove_self_loops(
    torch.LongTensor(np.load(args.edges)), edge_attr=edge_dates
)
edge_index = torch.cat((edge_index, edge_index[[1, 0]]), 1)
edge_dates = torch.cat((edge_dates, edge_dates))

node_classes = torch.LongTensor(np.load(args.node_classes))

edge_types = torch.sum(
    edge_index >= (len(node_classes) - torch.sum(node_classes)), 0
)
edge_types += torch.logical_and(edge_types == 2, edge_index[0] > edge_index[1])
features = torch.FloatTensor(np.load(args.features))

# sanity check
assert edge_index.shape[0] == 2
assert torch.max(edge_index) < node_classes.shape[0]

# train-test split edges
pos_train = edge_index[:, edge_dates <= args.test_year]
pos_val = edge_index[
    :,
    torch.logical_and(
        edge_dates > args.val_year, edge_dates <= args.test_year
    ),
]
pos_test = edge_index[:, edge_dates > args.test_year]
# sparse tensor
train_adj_t = SparseTensor(
    row=pos_train[0],
    col=pos_train[1],
    value=torch.ones(pos_train.shape[1], dtype=torch.float),
    sparse_sizes=(len(node_classes), len(node_classes)),
)
adj_t = SparseTensor(
    row=edge_index[0],
    col=edge_index[1],
    value=torch.ones(edge_index.shape[1], dtype=torch.float),
    sparse_sizes=(len(node_classes), len(node_classes)),
)
neg_train = negative_sampling(
    pos_train,
    num_nodes=len(node_classes),
    num_neg_samples=pos_train.shape[1] * 50,
    force_undirected=True,
)
neg_train = neg_train[:, torch.sum(neg_train >= len(node_classes), 0) == 0]
neg_edge_type = torch.sum(
    neg_train >= (len(node_classes) - torch.sum(node_classes)), 0
)
neg_train = neg_train[:, neg_edge_type == 1]
neg_train = neg_train[:, : pos_train.size(1)]

neg_val = negative_sampling(
    torch.cat((pos_val, pos_train), 1),
    num_nodes=len(node_classes),
    num_neg_samples=(pos_train.shape[1] + pos_val.shape[1]) * 50,
    force_undirected=True,
)
neg_val = neg_val[:, torch.sum(neg_val >= len(node_classes), 0) == 0]
neg_edge_type = torch.sum(
    neg_val >= (len(node_classes) - torch.sum(node_classes)), 0
)
neg_val = neg_val[:, neg_edge_type == 1]
neg_val = neg_val[:, : pos_val.size(1)]

neg_test = negative_sampling(
    edge_index,
    num_nodes=len(node_classes),
    num_neg_samples=edge_index.shape[1] * 50,
    force_undirected=True,
)
neg_test = neg_test[:, torch.sum(neg_test >= len(node_classes), 0) == 0]
neg_edge_type = torch.sum(
    neg_test >= (len(node_classes) - torch.sum(node_classes)), 0
)
neg_test = neg_test[:, neg_edge_type == 1]
neg_test = neg_test[:, : pos_test.size(1)]

# edge types
train_edge_types = (
    (
        train_adj_t.storage.row()
        > (len(node_classes) - torch.sum(node_classes))
    ).long()
    + (
        train_adj_t.storage.col()
        > (len(node_classes) - torch.sum(node_classes))
    ).long()
)
train_edge_types += torch.logical_and(
    train_adj_t.storage.row() < train_adj_t.storage.col(),
    train_edge_types == 2,
)
edge_types = (
    (
        adj_t.storage.row() > (len(node_classes) - torch.sum(node_classes))
    ).long()
    + (
        adj_t.storage.col() > (len(node_classes) - torch.sum(node_classes))
    ).long()
)
edge_types += torch.logical_and(
    adj_t.storage.row() < adj_t.storage.col(), edge_types == 2,
)

full_graph = gdata.Data(
    adj_t=adj_t,
    train_adj_t=train_adj_t,
    edge_types=edge_types,
    train_edge_types=train_edge_types,
    pos_val=pos_val,
    neg_val=neg_val,
    pos_train=pos_train,
    neg_train=neg_train,
    feats=features,
    node_classes=node_classes,
    num_nodes=len(node_classes),
)

mlflow.set_tracking_uri("http://localhost:12345")

with mlflow.start_run():
    # split stats
    mlflow.log_metric("train pos edges", len(pos_train[0]))
    mlflow.log_metric("val pos edges", len(pos_val[0]))
    mlflow.log_metric("test pos edges", len(pos_test[0]))
    mlflow.log_metric(
        "train POS/NEG", float(len(pos_train[0])) / len(neg_train[0])
    )
    mlflow.log_metric("val POS/NEG", float(len(pos_val[0])) / len(neg_val[0]))
    mlflow.log_metric(
        "test POS/NEG", float(len(pos_test[0])) / len(neg_test[0])
    )

    model = gnn.GAE(gae.CTDEncoder(62, args.dim, torch.sum(node_classes)))
    optimizer = opt.AdamW(model.parameters(), args.lr, weight_decay=args.wd)
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=100, verbose=True
    )
    model = gae.train_ctd_gae(
        model,
        [full_graph],
        optimizer,
        scheduler,
        args.device,
        args.epochs,
        callback,
    )
    torch.save(model, "./model.pt")

    model = torch.load("./best_model.pt")
    mlflow.pytorch.log_model(
        model,
        "unsupervised_ctd_model.pt",
        conda_env="conda.yaml",
        code_paths=["./"],
    )

    # full graph testing
    val_adj_t = SparseTensor(
        row=edge_index[0, edge_dates <= args.test_year],
        col=edge_index[1, edge_dates <= args.test_year],
        value=torch.ones(
            (edge_dates <= args.test_year).sum(), dtype=torch.float
        ),
        sparse_sizes=(len(node_classes), len(node_classes)),
    )
    val_edge_types = (
        (
            val_adj_t.storage.row()
            > (len(node_classes) - torch.sum(node_classes))
        ).long()
        + (
            val_adj_t.storage.col()
            > (len(node_classes) - torch.sum(node_classes))
        ).long()
    )
    val_edge_types += torch.logical_and(
        val_adj_t.storage.row() < val_adj_t.storage.col(), val_edge_types == 2,
    )
    model.eval()
    with torch.no_grad():
        z = model.encode(
            full_graph.feats.to(args.device),
            val_adj_t.to(args.device),
            val_edge_types,
        )
        auc, ap = model.test(
            z, pos_test.to(args.device), neg_test.to(args.device),
        )
        mlflow.log_metric("Chosen model test AUC GD", auc)
        mlflow.log_metric("Chosen model test AP GD", ap)
        z = model.encode(
            full_graph.feats.to(args.device),
            full_graph.train_adj_t.to(args.device),
            full_graph.train_edge_types,
        )
        auc, ap = model.test(
            z,
            full_graph.pos_val.to(args.device),
            full_graph.neg_val.to(args.device),
        )
        mlflow.log_metric("Chosen model val AUC GD", auc)
        mlflow.log_metric("Chosen model val AP GD", ap)

    # encode
    embeddings = gae.encode_ctd(model, full_graph, args.device)
    np.save("./embedding_unsupervised.npy", embeddings)
    mlflow.log_artifact("./embedding_unsupervised.npy")
