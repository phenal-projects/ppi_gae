from sys import argv
import argparse

import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch

import torch
import torch.optim as opt
import torch_geometric.nn as gnn
from torch_geometric.utils import train_test_split_edges

import data
import gae
from eval import class_test


best_auc = 0
counter = 0


def callback(model, auc_ap):
    global best_auc
    global counter
    auc, ap = auc_ap
    if best_auc < auc:
        best_auc = auc
        counter = 0
        torch.save(model, "./best_model.pt")
    counter += 1
    mlflow.log_metric("ROC_AUC", auc)
    mlflow.log_metric("AP", ap)
    if counter > 500:
        print("Stop!")
        return True
    return False


# Parameters
parser = argparse.ArgumentParser(description="Train GAE.")
parser.add_argument("lr", type=float, help="learning rate")
parser.add_argument("wd", type=float, help="weight decay")
parser.add_argument("epochs", type=int, help="the number of epochs to train")
parser.add_argument(
    "cut", type=float, help="the threshold for cutting weak edges"
)
parser.add_argument("dim", type=int, help="the size of embeddings")
parser.add_argument(
    "tissue", type=str, help="the column with expression in feats file",
)
parser.add_argument("device", type=str, help="cuda or cpu")

# Paths
parser.add_argument("edges", type=str, help="a path to the edge list (tsv)")
parser.add_argument("ids", type=str, help="a path to the ids tsv")
parser.add_argument(
    "feats", type=str, help="a path to the node features (tsv)"
)

args = parser.parse_args(argv[1:])

tracking_uri = "http://localhost:12345"
mlflow.set_tracking_uri(tracking_uri)

with mlflow.start_run():
    # logging
    mlflow.log_param("lr", args.lr)
    mlflow.log_param("wd", args.wd)
    mlflow.log_param("cut", args.cut)
    mlflow.log_param("dim", args.dim)
    mlflow.log_param("tissue", args.tissue)
    # data loading
    full_graph = data.graph_data(
        args.edges, args.feats, args.ids, cut=args.cut, sparse_tensor=False
    )

    ids = pd.read_csv(args.ids, sep="\t")
    expression = pd.read_csv(args.feats, sep="\t")
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

    # make sparse tensors
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
    mlflow.log_artifact(
        "./embedding_unsupervised.npy", "embedding_unsupervised.npy"
    )

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
        mlflow.log_metric("embedding_auc", auc)
        mlflow.log_metric("embedding_ap", ap)
