"""CTD Gene-Disease GAE training"""
from sys import argv
import argparse

import numpy as np

import torch
import torch.optim as opt
import torch_geometric.nn as gnn
import torch_geometric.data as gdata
from torch_geometric import utils
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
    parser.add_argument("epochs", type=int, help="the number of epochs to train")
    parser.add_argument("dim", type=int, help="the size of embeddings")
    parser.add_argument("device", type=str, help="cuda or cpu")
    parser.add_argument("seed", type=int, help="random seed for repruduction")

    # Paths
    parser.add_argument("edges", type=str, help="a path to the edge list (npy)")
    parser.add_argument(
        "features", type=str, help="a path to the features of the protein nodes (npy)",
    )
    parser.add_argument(
        "node_classes", type=str, help="a path to the expressions data file (npy)",
    )
    return parser


def callback(model, auc_ap_loss):
    """A callback function for printing training stats"""
    global BEST_AUC
    global ESTOP_COUNTER
    global EPOCH_COUNTER
    auc_gd, ap_gd, auc_gg, ap_gg, loss = auc_ap_loss
    if BEST_AUC < (auc_gd + auc_gg) / 2:
        BEST_AUC = (auc_gd + auc_gg) / 2
        ESTOP_COUNTER = 0
        torch.save(model, "./best_model.pt")
    ESTOP_COUNTER += 1
    mlflow.log_metric("ROC_AUC_GD", auc_gd, step=EPOCH_COUNTER)
    mlflow.log_metric("AP_GD", ap_gd, step=EPOCH_COUNTER)
    mlflow.log_metric("ROC_AUC_GG", auc_gg, step=EPOCH_COUNTER)
    mlflow.log_metric("AP_GG", ap_gg, step=EPOCH_COUNTER)
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
edge_index = utils.remove_self_loops(torch.LongTensor(np.load(args.edges)))[0]
node_classes = torch.LongTensor(np.load(args.node_classes))
edge_types = torch.sum(edge_index >= (len(node_classes) - torch.sum(node_classes)), 0)
features = torch.FloatTensor(np.load(args.features))


# sanity check
assert edge_index.shape[0] == 2
assert torch.max(edge_index) < node_classes.shape[0]

# train-test split edges
genes = torch.arange(len(node_classes))[node_classes == 0]
diseases = torch.arange(len(node_classes))[node_classes == 1]
validation_genes_mask = torch.randint(
    0, 100, size=(len(node_classes) - torch.sum(node_classes).item(),)
)
validation_genes = torch.arange(
    0, len(node_classes) - torch.sum(node_classes), dtype=torch.long
)[validation_genes_mask < 35]

full_graph = gdata.Data(
    edge_index=torch.cat((edge_index, edge_index[[1, 0]]), 1),
    edge_types=torch.cat((edge_types, edge_types)),
    feats=features,
    node_classes=node_classes,
    num_nodes=len(node_classes),
)

# positive train/val
pos_val = torch.logical_or(
    torch.logical_and(
        torch.BoolTensor(np.isin(full_graph.edge_index[0], validation_genes),),
        full_graph.edge_types == 1,
    ),
    torch.logical_and(
        torch.BoolTensor(np.isin(full_graph.edge_index[1], validation_genes),),
        full_graph.edge_types == 1,
    ),
)
full_graph.pos_val_gd = full_graph.edge_index[:, pos_val]
full_graph.pos_train_gd = full_graph.edge_index[
    :, torch.logical_and(torch.logical_not(pos_val), full_graph.edge_types == 1),
]
pos_val = torch.logical_and(
    torch.randint(0, 3, size=(len(full_graph.edge_types),)) == 0,
    full_graph.edge_types == 0,
)
full_graph.pos_val_gg = full_graph.edge_index[:, pos_val]
full_graph.pos_train_gg = full_graph.edge_index[
    :, torch.logical_and(torch.logical_not(pos_val), full_graph.edge_types == 0),
]

# negatives
# utils.negative_sampling sometimes produces edges with unexisting nodes. So here is a small patch.
neg = utils.negative_sampling(
    full_graph.edge_index, num_nodes=full_graph.num_nodes, force_undirected=True
)
neg = neg[:, torch.sum(neg >= len(node_classes), 0) == 0]  # PATCH
neg_edge_type = torch.sum(neg >= (len(node_classes) - torch.sum(node_classes)), 0)
neg_val = torch.logical_or(
    torch.logical_and(
        torch.BoolTensor(np.isin(neg[0], validation_genes),), neg_edge_type == 1,
    ),
    torch.logical_and(
        torch.BoolTensor(np.isin(neg[1], validation_genes),), neg_edge_type == 1,
    ),
)
full_graph.neg_val_gd = neg[:, neg_val]
full_graph.neg_train_gd = neg[
    :, torch.logical_and(torch.logical_not(neg_val), neg_edge_type == 1)
]
neg_val = torch.logical_and(
    torch.randint(0, 3, size=(neg.shape[1],)) == 0, neg_edge_type == 0,
)
full_graph.neg_val_gg = neg[:, neg_val]
full_graph.neg_train_gg = neg[
    :, torch.logical_and(torch.logical_not(neg_val), neg_edge_type == 0)
]

# sparse tensor
full_graph.train_adj_t_gg = SparseTensor(
    row=full_graph.pos_train_gg[0],
    col=full_graph.pos_train_gg[1],
    value=torch.ones(full_graph.pos_train_gg.shape[1], dtype=torch.float),
    sparse_sizes=(len(node_classes), len(node_classes)),
)
full_graph.train_adj_t_gd = SparseTensor(
    row=full_graph.pos_train_gd[0],
    col=full_graph.pos_train_gd[1],
    value=torch.ones(full_graph.pos_train_gd.shape[1], dtype=torch.float),
    sparse_sizes=(len(node_classes), len(node_classes)),
)
full_graph.adj_t_gg = SparseTensor(
    row=full_graph.edge_index[0][full_graph.edge_types == 0],
    col=full_graph.edge_index[1][full_graph.edge_types == 0],
    value=torch.ones(torch.sum(full_graph.edge_types == 0).item(), dtype=torch.float),
    sparse_sizes=(len(node_classes), len(node_classes)),
)
full_graph.adj_t_gd = SparseTensor(
    row=full_graph.edge_index[0][full_graph.edge_types == 1],
    col=full_graph.edge_index[1][full_graph.edge_types == 1],
    value=torch.ones(torch.sum(full_graph.edge_types == 1).item(), dtype=torch.float),
    sparse_sizes=(len(node_classes), len(node_classes)),
)
full_graph.adj_t_dd = SparseTensor(
    row=full_graph.edge_index[0][full_graph.edge_types == 2],
    col=full_graph.edge_index[1][full_graph.edge_types == 2],
    value=torch.ones(torch.sum(full_graph.edge_types == 2).item(), dtype=torch.float),
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
        model, [full_graph], optimizer, scheduler, args.device, args.epochs, callback,
    )
    torch.save(model, "./model.pt")

    model = torch.load("./best_model.pt")
    mlflow.pytorch.log_model(
        model, "unsupervised_ctd_model.pt", conda_env="conda.yaml", code_paths=["./"],
    )

    # full graph testing
    model.eval()
    with torch.no_grad():
        z = model.encode(
            full_graph.feats.to(args.device),
            full_graph.adj_t_gg.to(args.device),
            full_graph.adj_t_gd.to(args.device),
            full_graph.adj_t_dd.to(args.device),
        )
        auc, ap = model.test(
            z,
            full_graph.pos_val_gd.to(args.device),
            full_graph.neg_val_gd.to(args.device),
        )
    mlflow.log_metric("Final AUC GD", auc)
    mlflow.log_metric("Final AP GD", ap)

    # encode
    embeddings = gae.encode_ctd(model, full_graph, args.device)
    np.save("./embedding_unsupervised.npy", embeddings)
    mlflow.log_artifact("./embedding_unsupervised.npy")
    torch.save(validation_genes, "val_node.pt")
    mlflow.log_artifact("./val_node.pt")
