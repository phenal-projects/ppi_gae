"""Finetuning encoder with classification head"""
from sys import argv
import argparse

import numpy as np
import pandas as pd

import torch
import torch.optim as opt
import torch_geometric.nn as gnn
from torch_geometric.utils import train_test_split_edges

import mlflow
import mlflow.pytorch

import data
import gae
from eval import class_test


BEST_AUC = 0
ESTOP_COUNTER = 0
EPOCH_COUNTER = 0


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


# Parameter
def construct_parser():
    """Constructs args parser"""
    parser = argparse.ArgumentParser(description="Train GAE.")
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("wd", type=float, help="weight decay")
    parser.add_argument(
        "epochs", type=int, help="the number of epochs to train"
    )
    parser.add_argument(
        "cut", type=float, help="the threshold for cutting weak edges"
    )
    parser.add_argument("dim", type=int, help="the size of embeddings")
    parser.add_argument(
        "tissue",
        type=str,
        help="the column with expression in expression file",
    )
    parser.add_argument("device", type=str, help="cuda or cpu")
    parser.add_argument("seed", type=int, help="random seed for repruduction")

    # Paths
    parser.add_argument(
        "edges", type=str, help="a path to the edge list (tsv)"
    )
    parser.add_argument("ids", type=str, help="a path to the ids tsv")
    parser.add_argument(
        "feats", type=str, help="a path to the node features (tsv)"
    )
    parser.add_argument(
        "expression",
        type=str,
        help="a path to the expressions data file (tsv)",
    )
    return parser


def log_args(args):
    """Logs all necessary data"""
    mlflow.log_param("lr", args.lr)
    mlflow.log_param("wd", args.wd)
    mlflow.log_param("cut", args.cut)
    mlflow.log_param("dim", args.dim)
    mlflow.log_param("tissue", args.tissue)
    mlflow.log_param("seed", args.seed)


def load_data(args):
    """Loads graph data and splits the graph into clusters"""
    full_graph = data.graph_data(
        args.edges, args.feats, args.ids, cut=args.cut, sparse_tensor=False
    )

    ids = pd.read_csv(args.ids, sep="\t")
    expression = pd.read_csv(args.expression, sep="\t")
    expression = ids[["id", "ensembl.gene"]].merge(
        expression, left_on="ensembl.gene", right_on="Gene", how="left"
    )
    expression = expression.drop(["id", "ensembl.gene", "Gene"], 1).fillna(0)
    stds = expression.T.std()
    if args.tissue in expression.columns:
        full_graph = data.tissue_specific_ppi_cut(
            full_graph, expression[args.tissue] / stds
        )
    else:
        full_graph.new_id = full_graph.id
    loader = data.cluster_data(full_graph, 1, 1, shuffle=True, verbose=True)
    return loader, full_graph


def construct_sparse(loader):
    """Construct sparse matrices in the graphs from the loader"""
    graphs = []
    for graph in loader:
        graph.edge_id = dict(
            list(
                zip(
                    (
                        graph.num_nodes * graph.edge_index[0]
                        + graph.edge_index[1]
                    )
                    .numpy()
                    .squeeze(),
                    graph.edge_attr.squeeze().numpy(),
                )
            )
        )
        graphs.append(
            data.make_sparse(
                train_test_split_edges(graph, val_ratio=0.3, test_ratio=0)
            )
        )
    return graphs


parser = construct_parser()

args = parser.parse_args(argv[1:])

# deterministic behaviour
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

mlflow.set_tracking_uri("http://localhost:12345")

with mlflow.start_run():
    # logging
    log_args(args)
    # data loading
    loader, full_graph = load_data(args)

    # make sparse tensors
    graphs = construct_sparse(loader)

    # logging data stats
    labels = data.labels_data(args.ids, ["skin_integrity"])[
        full_graph.new_id.numpy()
    ]
    mlflow.log_param("nodes_left", full_graph.new_id.shape[0])
    mlflow.log_param("total_support", np.sum(labels))

    # model preparation
    model = gnn.GAE(gae.Encoder(62, args.dim))
    optimizer = opt.AdamW(
        model.parameters(), args.lr, weight_decay=args.wd, amsgrad=True
    )
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20, verbose=True
    )

    model = gae.train_gae(
        model, graphs, optimizer, scheduler, args.device, args.epochs, callback
    )
    torch.save(model, "./model.pt")

    model = torch.load("./best_model.pt")
    mlflow.pytorch.log_model(
        model,
        "unsupervised_model.pt",
        conda_env="conda.yaml",
        code_paths=["./"],
    )
    # encode
    embeddings = []
    ids = []
    for graph in graphs:
        embeddings.append(gae.encode(model, graph, args.device))
        ids.append(graph.id)
    ids = torch.cat(ids, 0)
    embeddings = np.concatenate(embeddings, 0)
    embeddings = embeddings[ids.argsort().numpy()]
    np.save("./embedding_unsupervised.npy", embeddings)
    mlflow.log_artifact("./embedding_unsupervised.npy")

    # classification test
    classes = [
        "Cancer-related genes",
        "Disease related genes",
        "Enzymes",
        "FDA approved drug targets",
        "G-protein coupled receptors",
        "Plasma proteins",
        "Potential drug targets",
        "Predicted intracellular proteins",
        "Predicted membrane proteins",
        "Predicted secreted proteins",
        "Transcription factors",
        "Transporters",
        "skin_integrity",
    ]

    classification_results = class_test(
        embeddings[full_graph.new_id.numpy()],
        data.labels_data(args.ids, classes)[full_graph.new_id.numpy()],
        method="auc",
    )
    for key in classification_results:
        auc = classification_results[key]["roc"]
        ap = classification_results[key]["ap"]
        mlflow.log_metric("auc_" + classes[key], auc)
        mlflow.log_metric("ap_" + classes[key], ap)
