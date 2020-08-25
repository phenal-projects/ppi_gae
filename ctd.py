"""CTD Gene-Disease GAE training"""
from sys import argv
import argparse

import numpy as np

import torch
import torch.optim as opt
import torch_geometric.nn as gnn
import torch_geometric.data as gdata
from torch_geometric.utils import (
    train_test_split_edges,
    remove_self_loops,
)
from torch_sparse import SparseTensor

import mlflow
import mlflow.pytorch

import gae


BEST_AUC = 0
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
    parser.add_argument("dim", type=int, help="the size of embeddings")
    parser.add_argument("device", type=str, help="cuda or cpu")
    parser.add_argument("seed", type=int, help="random seed for repruduction")

    # Paths
    parser.add_argument(
        "edges", type=str, help="a path to the edge list (npy)"
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


def callback(model, auc_ap):
    """A callback function for printing training stats"""
    global BEST_AUC
    global ESTOP_COUNTER
    global EPOCH_COUNTER
    auc, ap = auc_ap
    if BEST_AUC < auc:
        BEST_AUC = auc
        ESTOP_COUNTER = 0
        torch.save(model, "./best_model.pt")
    ESTOP_COUNTER += 1
    mlflow.log_metric("ROC_AUC", auc, step=EPOCH_COUNTER)
    mlflow.log_metric("AP", ap, step=EPOCH_COUNTER)
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
edge_index = torch.LongTensor(np.load(args.edges))
features = torch.FloatTensor(np.load(args.features))
node_classes = torch.LongTensor(np.load(args.node_classes))

# sanity check
assert edge_index.shape[0] == 2
assert torch.max(edge_index) < node_classes.shape[0]

# train-test split edges
genes = torch.arange(len(node_classes))[node_classes == 0]
diseases = torch.arange(len(node_classes))[node_classes == 1]
interclass_mask = torch.BoolTensor(
    np.logical_or(
        np.isin(edge_index[0], diseases), np.isin(edge_index[1], diseases),
    )
)
interclass_edges = edge_index[:, interclass_mask]

full_graph = gdata.Data(
    edge_index=remove_self_loops(
        torch.cat((interclass_edges, interclass_edges[[1, 0]]), 1),
    )[0],
    feats=features,
    node_classes=node_classes,
    num_nodes=len(node_classes),
)

full_graph = train_test_split_edges(full_graph, val_ratio=0.3, test_ratio=0)
full_graph.train_pos_with_ppi = torch.cat(
    (
        full_graph.train_pos_edge_index,
        edge_index[:, ~interclass_mask],
        edge_index[:, ~interclass_mask][[1, 0]],
    ),
    1,
)

# save split to check it later
torch.save(full_graph.train_pos_edge_index, "train_ei.pt")
torch.save(full_graph.val_pos_edge_index, "val_ei.pt")
torch.save(full_graph.val_neg_edge_index, "val_ei_neg.pt")

full_graph.train_adj_t = SparseTensor(
    row=full_graph.train_pos_with_ppi[0],
    col=full_graph.train_pos_with_ppi[1],
    value=torch.ones(
        len(full_graph.train_pos_with_ppi[1]), dtype=torch.float32
    ),
    sparse_sizes=(len(node_classes), len(node_classes)),
)
full_graph.adj_t = SparseTensor(
    row=torch.cat((edge_index[0], edge_index[1])),
    col=torch.cat((edge_index[1], edge_index[0])),
    value=torch.ones(2 * len(edge_index[1]), dtype=torch.float32),
    sparse_sizes=(len(node_classes), len(node_classes)),
)

mlflow.set_tracking_uri("http://localhost:12345")

with mlflow.start_run():
    model = gnn.GAE(gae.CTDEncoder(62, args.dim, torch.sum(node_classes)))
    optimizer = opt.AdamW(
        model.parameters(), args.lr, weight_decay=args.wd, amsgrad=True
    )
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20, verbose=True
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
    model.eval()
    with torch.no_grad():
        z = model.encode(
            full_graph.feats.to(args.device), full_graph.adj_t.to(args.device)
        )
        auc, ap = model.test(
            z,
            full_graph.val_pos_edge_index.to(args.device),
            full_graph.val_neg_edge_index.to(args.device),
        )
    mlflow.log_metric("Final AUC", auc)
    mlflow.log_metric("Final AP", ap)

    # encode
    embeddings = gae.encode_ctd(model, full_graph, args.device)
    np.save("./embedding_unsupervised.npy", embeddings)
    mlflow.log_artifact("./embedding_unsupervised.npy")
