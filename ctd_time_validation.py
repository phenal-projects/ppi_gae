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
ESTOP_COUNTER = 0
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


def neg_sample(pos_ei, num_nodes, min_dis_id, edge_type=1):
    pos_edge_types = torch.sum(pos_ei >= min_dis_id, 0)
    pos_edge_types += torch.logical_and(pos_edge_types == 2, pos_ei[0] > pos_ei[1])
    neg_sample_size = torch.sum(pos_edge_types == edge_type)
    neg_sample = negative_sampling(
        pos_ei,
        num_nodes=num_nodes,
        num_neg_samples=neg_sample_size * 25,  # oversampling
        force_undirected=True,
    )
    neg_sample = neg_sample[:, torch.sum(neg_sample > num_nodes, 0) == 0]  # quick fix
    neg_edge_type = torch.sum(neg_sample >= min_dis_id, 0)
    neg_edge_type += torch.logical_and(
        neg_edge_type == 2, neg_sample[0] > neg_sample[1]
    )
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
    min_dis_id = len(node_classes) - torch.sum(node_classes)
    edge_types = torch.sum(edge_index >= min_dis_id, 0)
    edge_types += torch.logical_and(edge_types == 2, edge_index[0] > edge_index[1])
    features = torch.FloatTensor(np.load(args.features))

    # sanity check
    assert edge_index.shape[0] == 2
    assert torch.max(edge_index) < node_classes.shape[0]

    # train-test split edges
    pos_train_gd = edge_index[
        :,
        torch.logical_and(
            torch.logical_and(edge_dates < args.val_year, edge_types == 1),
            edge_index[0] < edge_index[1],
        ),
    ]
    pos_val_gd = edge_index[
        :,
        torch.logical_and(
            torch.logical_and(
                torch.logical_and(
                    edge_dates >= args.val_year, edge_dates < args.test_year
                ),
                edge_types == 1,
            ),
            edge_index[0] < edge_index[1],
        ),
    ]
    pos_test_gd = edge_index[
        :,
        torch.logical_and(
            torch.logical_and(edge_dates >= args.test_year, edge_types == 1),
            edge_index[0] < edge_index[1],
        ),
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
    train_edge_types = (train_adj_t.storage.row() >= min_dis_id).long() + (
        train_adj_t.storage.col() >= min_dis_id
    ).long()
    train_edge_types += torch.logical_and(
        train_adj_t.storage.row() < train_adj_t.storage.col(), train_edge_types == 2,
    )
    edge_types = (adj_t.storage.row() >= min_dis_id).long() + (
        adj_t.storage.col() >= min_dis_id
    ).long()
    edge_types += torch.logical_and(
        adj_t.storage.row() < adj_t.storage.col(), edge_types == 2,
    )

    # negatives
    neg = neg_sample(edge_index, len(node_classes), min_dis_id)
    neg_train_gd = neg[:, : torch.sum(train_edge_types == 1)]
    neg_val_gd = neg[
        :,
        torch.sum(train_edge_types == 1) : torch.sum(train_edge_types == 1)
        + pos_val_gd.size(-1),
    ]
    neg_test_gd = neg[
        :, torch.sum(train_edge_types == 1) + pos_val_gd.size(-1) :,
    ]

    full_graph = gdata.Data(
        adj_t=adj_t,
        train_adj_t=train_adj_t,
        edge_types=edge_types,
        train_edge_types=train_edge_types,
        pos_val_gd=pos_val_gd,
        neg_val_gd=neg_val_gd,
        pos_train_gd=pos_train_gd,
        neg_train_gd=neg_train_gd,
        feats=features,
        node_classes=node_classes,
        num_nodes=len(node_classes),
    )

    mlflow.set_tracking_uri("http://localhost:12345")

    with mlflow.start_run():
        # split stats
        mlflow.log_metric(
            "train pos gene-disease edges", torch.sum(train_edge_types == 1).item()
        )
        mlflow.log_metric("val pos gene-disease edges", len(pos_val_gd[0]))
        mlflow.log_metric("test pos gene-disease edges", len(pos_test_gd[0]))
        mlflow.log_metric(
            "train POS/NEG",
            float(torch.sum(train_edge_types == 1)) / len(neg_train_gd[0]),
        )
        mlflow.log_metric("val POS/NEG", float(len(pos_val_gd[0])) / len(neg_val_gd[0]))
        mlflow.log_metric(
            "test POS/NEG", float(len(pos_test_gd[0])) / len(neg_test_gd[0])
        )

        model = gnn.GAE(
            gae.CTDEncoder(62, args.dim, torch.sum(node_classes)),
            gae.RelDecoder(args.dim, 2, 16),
        )
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

        # model = torch.load("./best_model.pt")
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
        val_edge_types = (val_adj_t.storage.row() >= min_dis_id).long() + (
            val_adj_t.storage.col() >= min_dis_id
        ).long()
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
            auc, ap = gae.test(
                z,
                model.decoder,
                (0, 1),
                pos_test_gd.to(args.device),
                neg_test_gd.to(args.device),
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
                (0, 1),
                full_graph.pos_val_gd.to(args.device),
                full_graph.neg_val_gd.to(args.device),
            )
            mlflow.log_metric("Chosen model val AUC GD", auc)
            mlflow.log_metric("Chosen model val AP GD", ap)

        # encode
        embeddings = gae.encode_ctd(model, full_graph, args.device)
        np.save("./embedding_unsupervised.npy", embeddings)
        mlflow.log_artifact("./embedding_unsupervised.npy")
