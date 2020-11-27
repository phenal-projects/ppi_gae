"""CTD Gene-Disease GAE training"""
import argparse
from sys import argv

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.optim as opt
import torch_geometric.data as gdata
import torch_geometric.nn as gnn
from torch_geometric import utils
from torch_geometric.utils.negative_sampling import negative_sampling
from torch_sparse import SparseTensor

import gae

BEST_AP = 0
EPOCH_COUNTER = 0


def construct_parser():
    """Constructs args parser"""
    parser = argparse.ArgumentParser(description="Train HGAE.")
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("wd", type=float, help="weight decay")
    parser.add_argument("epochs", type=int, help="the number of epochs to train")
    parser.add_argument(
        "val_year", type=int, help="after this year the edges will be considered val",
    )
    parser.add_argument(
        "test_year", type=int, help="after this year the edges will be considered test",
    )
    parser.add_argument("dim", type=int, help="the size of embeddings")
    parser.add_argument("device", type=str, help="cuda or cpu")
    parser.add_argument("seed", type=int, help="random seed for repruduction")

    # Paths
    parser.add_argument("edges", type=str, help="a path to the edge list (npy)")
    parser.add_argument(
        "edges_dates", type=str, help="a list of years for train/val/test split",
    )
    parser.add_argument(
        "features", type=str, help="a path to the features of the protein nodes (npy)",
    )
    parser.add_argument(
        "node_classes", type=str, help="a path to the expressions data file (npy)",
    )
    return parser


def callback(model, auc_ap_loss):
    """A callback function for printing training stats"""
    global BEST_AP
    global EPOCH_COUNTER
    auc_gd, ap_gd, loss = auc_ap_loss
    if BEST_AP < ap_gd and EPOCH_COUNTER > 100:
        BEST_AP = ap_gd
        torch.save(model, "./best_model.pt")
    mlflow.log_metric("ROC_AUC_GD", auc_gd, step=EPOCH_COUNTER)
    mlflow.log_metric("AP_GD", ap_gd, step=EPOCH_COUNTER)
    mlflow.log_metric("LOSS", loss, step=EPOCH_COUNTER)
    EPOCH_COUNTER += 1
    return False


def neg_sample(pos_ei, num_nodes, node_classes, edge_type=1):
    """edge_type is calculated as class_from * node_classes + class_to"""
    num_classes = node_classes.max() + 1
    neg_sample_size = pos_ei.shape[-1]
    neg_sample = negative_sampling(
        pos_ei,
        num_nodes=num_nodes,
        num_neg_samples=neg_sample_size * (num_classes ** 2 * 3),  # oversampling
        force_undirected=True,
    )
    neg_sample = neg_sample[:, torch.sum(neg_sample >= num_nodes, 0) == 0]  # quick fix
    neg_classes = node_classes[neg_sample]
    neg_edge_type = neg_classes[0] * num_classes + neg_classes[1]
    neg_sample = neg_sample[:, neg_edge_type == edge_type]
    neg_sample = neg_sample[:, :neg_sample_size]
    return neg_sample


if __name__ == "__main__":
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
    num_classes = node_classes.max() + 1
    classes = node_classes[edge_index]
    edge_types = classes[0] * num_classes + classes[1]
    features = torch.FloatTensor(np.load(args.features))

    # sanity check
    assert edge_index.shape[0] == 2
    assert torch.max(edge_index) < node_classes.shape[0]

    # train-test split edges
    pos_train = {}
    pos_val = {}
    pos_test = {}
    neg_test = {}
    neg_val = {}
    for edge_type in edge_types.unique():
        pos_train[edge_type.item()] = edge_index[
            :, torch.logical_and(edge_dates < args.val_year, edge_types == edge_type),
        ]
        pos_val[edge_type.item()] = edge_index[
            :,
            torch.logical_and(
                torch.logical_and(
                    edge_dates >= args.val_year, edge_dates < args.test_year
                ),
                edge_types == edge_type,
            ),
        ]
        pos_test[edge_type.item()] = edge_index[
            :, torch.logical_and(edge_dates >= args.test_year, edge_types == edge_type),
        ]
        neg = neg_sample(
            edge_index, len(node_classes), node_classes, edge_type=edge_type.item()
        )
        neg_val[edge_type.item()] = neg[
            :,
            pos_train[edge_type.item()].size(-1) : pos_train[edge_type.item()].size(-1)
            + pos_val[edge_type.item()].size(-1),
        ]
        neg_test[edge_type.item()] = neg[
            :,
            pos_train[edge_type.item()].size(-1)
            + pos_val[edge_type.item()].size(-1) : pos_train[edge_type.item()].size(-1)
            + pos_val[edge_type.item()].size(-1)
            + pos_test[edge_type.item()].size(-1),
        ]

    # sparse tensor
    train_adj_t = SparseTensor(
        row=edge_index[0, edge_dates < args.val_year],
        col=edge_index[1, edge_dates < args.val_year],
        value=torch.ones(
            edge_index[0, edge_dates < args.val_year].shape[0], dtype=torch.float
        ),
        sparse_sizes=(len(node_classes), len(node_classes)),
    )
    adj_t = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=torch.ones(edge_index.shape[1], dtype=torch.float),
        sparse_sizes=(len(node_classes), len(node_classes)),
    )

    # edge types
    train_edge_types = (
        node_classes[train_adj_t.storage.row()] * num_classes
        + node_classes[train_adj_t.storage.col()]
    )
    edge_types = (
        node_classes[adj_t.storage.row()] * num_classes
        + node_classes[adj_t.storage.col()]
    )

    full_graph = gdata.Data(
        adj_t=adj_t,
        train_adj_t=train_adj_t,
        edge_types=edge_types,
        train_edge_types=train_edge_types,
        pos_val=pos_val,
        neg_val=neg_val,
        pos_train=pos_train,
        feats=features,
        node_classes=node_classes,
        num_nodes=len(node_classes),
        loss_weights=[
            1 if (x != 1 and x != 3) else 1.5 for x in range(num_classes ** 2)
        ],
    )

    mlflow.set_tracking_uri("http://localhost:12345")

    with mlflow.start_run():
        # split stats
        for edge_type in edge_types.unique():
            mlflow.log_param(
                "train pos edges - type " + str(edge_type.item()),
                len(pos_train[edge_type.item()][0]),
            )
            mlflow.log_param(
                "val pos edges - type " + str(edge_type.item()),
                len(pos_val[edge_type.item()][0]),
            )
            mlflow.log_param(
                "test pos edges - type " + str(edge_type.item()),
                len(pos_test[edge_type.item()][0]),
            )

        model = gnn.GAE(
            gae.CTDEncoder(62, args.dim, torch.sum(node_classes != 0)),
            gae.RelDecoder(args.dim, num_classes ** 2),
        )
        mlflow.log_param(
            "model parameters", sum([p.numel() for p in model.parameters()])
        )
        optimizer = opt.SGD(
            model.parameters(), args.lr, momentum=0.9, weight_decay=args.wd
        )
        scheduler = opt.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=200, T_mult=2, eta_min=1e-7
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
            row=edge_index[0, edge_dates < args.test_year],
            col=edge_index[1, edge_dates < args.test_year],
            value=torch.ones((edge_dates < args.test_year).sum(), dtype=torch.float),
            sparse_sizes=(len(node_classes), len(node_classes)),
        )
        val_edge_types = (
            node_classes[val_adj_t.storage.row()] * num_classes
            + node_classes[val_adj_t.storage.col()]
        )
        model.eval()
        with torch.no_grad():
            z = model.encode(
                full_graph.feats.to(args.device),
                val_adj_t.to(args.device),
                val_edge_types,
            )
            auc, ap = gae.test(
                z,
                model.decoder,
                1,
                pos_test[1].to(args.device),
                neg_test[1].to(args.device),
            )
            mlflow.log_metric("Chosen model test AUC GD", auc)
            mlflow.log_metric("Chosen model test AP GD", ap)
            z = model.encode(
                full_graph.feats.to(args.device),
                full_graph.train_adj_t.to(args.device),
                full_graph.train_edge_types,
            )
            auc, ap = gae.test(
                z,
                model.decoder,
                1,
                full_graph.pos_val[1].to(args.device),
                full_graph.neg_val[1].to(args.device),
            )
            mlflow.log_metric("Chosen model val AUC GD", auc)
            mlflow.log_metric("Chosen model val AP GD", ap)

        # encode
        embeddings = gae.encode_ctd(model, full_graph, args.device)
        np.save("./embedding_unsupervised.npy", embeddings)
        mlflow.log_artifact("./embedding_unsupervised.npy")
