"""Unsupervised model training"""
from sys import argv
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.optim as opt
import torch.nn as nn
from torch_geometric.utils import train_test_split_edges

import mlflow
import mlflow.pytorch

import data
import gae
import sim_pu
from eval import class_test


class Classification(nn.Module):
    def __init__(self, encoder, n_classes):
        super(Classification, self).__init__()
        self.encoder = encoder
        self.lin = nn.Linear(encoder.conv3.out_channels * 5, n_classes)

    def forward(self, x, adj):
        return self.lin(self.encoder(x, adj))


BEST_LOSS = float("inf")
ESTOP_COUNTER = 0
EPOCH_COUNTER = 0


def callback(model, loss):
    """A callback function for printing training stats"""
    global BEST_LOSS
    global ESTOP_COUNTER
    global EPOCH_COUNTER
    if BEST_LOSS > loss:
        BEST_LOSS = loss
        ESTOP_COUNTER = 0
        torch.save(model, "./best_finetuned_model.pt")
    ESTOP_COUNTER += 1
    mlflow.log_metric("loss", loss, step=EPOCH_COUNTER)
    EPOCH_COUNTER += 1
    if ESTOP_COUNTER > 500:
        print("Stop!")
        return True
    return False


def construct_parser():
    """Constructs args parser"""
    parser = argparse.ArgumentParser(description="Finetune pretrained model.")
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("wd", type=float, help="weight decay")
    parser.add_argument(
        "epochs", type=int, help="the number of epochs to train"
    )
    parser.add_argument(
        "k", type=int, help="the number of neighbors to enrich positive class"
    )
    parser.add_argument(
        "cut", type=float, help="the threshold for cutting weak edges"
    )
    parser.add_argument(
        "target", type=str, help="a column in ids file to be used as a target"
    )
    parser.add_argument(
        "tissue", type=str, help="column with expression in expression file"
    )
    parser.add_argument("device", type=str, help="cuda or cpu")
    parser.add_argument("seed", type=int, help="random seed for repruduction")

    # Paths
    parser.add_argument("model", type=str, help="a path to a serialized model")
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
    mlflow.log_param("k", args.k)
    mlflow.log_param("cut", args.cut)
    mlflow.log_param("target", args.target)
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
    target = data.labels_data(args.ids, [args.target])
    full_graph.y = torch.tensor(
        target[full_graph["id"]].squeeze(), dtype=torch.long
    )
    full_graph.train_nodes_mask = torch.rand((full_graph.num_nodes,)) > 0.3
    loader = data.cluster_data(full_graph, 1, 1, shuffle=True, verbose=True)
    return loader, target, full_graph


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
                train_test_split_edges(graph, val_ratio=0, test_ratio=0)
            )
        )
    return graphs


def get_embeddings(model):
    """Generates embeddings from the provided model"""
    embeddings = []
    ids = []
    for graph in graphs:
        embeddings.append(gae.encode(model, graph, args.device))
        ids.append(graph.id)
    ids = torch.cat(ids, 0)
    embeddings = np.concatenate(embeddings, 0)
    embeddings = embeddings[ids.argsort().numpy()]
    return embeddings


def probtagging(full_graph, graphs, embeddings, target, k):
    """Constructs probabilities for PU learning"""
    probs = full_graph.y.float()
    probs[full_graph.train_nodes_mask] = torch.tensor(
        sim_pu.knn_prob(
            embeddings[full_graph.train_nodes_mask],
            target[full_graph.train_nodes_mask],
            k,
        ),
        dtype=torch.float,
    )
    for graph in graphs:
        graph.probs = torch.tensor(probs[graph.id])
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
    loader, target, full_graph = load_data(args)

    # make sparse tensors
    graphs = construct_sparse(loader)

    # model preparation
    mdl = torch.load(args.model)
    model = Classification(mdl.encoder, 2)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(
            1
            / np.unique(target, return_counts=True)[1]
            / np.sum(1 / np.unique(target, return_counts=True)[1]),
            dtype=torch.float32,
            device=torch.device(args.device),
        )
    )

    # encode
    embeddings = get_embeddings(mdl)
    np.save("./embedding_unsupervised.npy", embeddings)
    # probtagging
    if args.k < 0:
        ks, expc = sim_pu.elbow_curve(embeddings, target)
        plt.plot(ks, expc)
        plt.savefig("elbow_plot.png")
        print("What k should be used? (see elbow_plot.png)")

        args.k = int(input())

    graphs = probtagging(full_graph, graphs, embeddings, target, args.k)

    # pretune linear layer
    # freeze all layers but the last linear one
    for param in model.parameters():
        param.requires_grad = False
    for param in model.lin.parameters():
        param.requires_grad = True

    optimizer = opt.AdamW(
        model.parameters(), 0.005, weight_decay=args.wd, amsgrad=True
    )
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20, verbose=True
    )
    model = gae.finetune_gae(
        model,
        graphs,
        loss_fn,
        optimizer,
        scheduler,
        args.device,
        200,
        callback,
    )

    # finetune
    # unfreeze
    for era in range(20):
        for param in model.parameters():
            param.requires_grad = True
        optimizer = opt.AdamW(
            model.parameters(), args.lr, weight_decay=args.wd, amsgrad=True
        )
        scheduler = opt.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=20, verbose=True
        )
        model = gae.finetune_gae(
            model,
            graphs,
            loss_fn,
            optimizer,
            scheduler,
            args.device,
            args.epochs // 20,
            callback,
        )
        torch.save(model, "./finetuned_model.pt")

        model = torch.load("./best_finetuned_model.pt")
        mlflow.pytorch.log_model(
            model,
            "finetuned_model.pt",
            conda_env="conda.yaml",
            code_paths=["./"],
        )

        # encode
        embeddings = get_embeddings(mdl)
        np.save("./embedding_finetuned.npy", embeddings)
        mlflow.log_artifact("./embedding_finetuned.npy")

        # retagging
        graphs = probtagging(full_graph, graphs, embeddings, target, args.k)

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
        val_mask=~full_graph.train_nodes_mask.numpy()[
            full_graph.new_id.numpy()
        ],
        method="auc",
        enrichment=args.k,
    )
    for key in classification_results:
        mlflow.log_metric(
            "auc_" + classes[key], classification_results[key]["roc"]
        )
        mlflow.log_metric(
            "ap_" + classes[key], classification_results[key]["ap"]
        )
        mlflow.log_metric(
            "auc_pval_" + classes[key], classification_results[key]["roc_pval"]
        )
        mlflow.log_metric(
            "ap_pval_" + classes[key], classification_results[key]["ap_pval"]
        )
