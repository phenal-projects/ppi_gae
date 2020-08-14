"""Finetuning encoder with classification head"""
from sys import argv
import argparse

import numpy as np

import torch
import torch.optim as opt
import torch_geometric.nn as gnn
import torch_geometric.data as gdata
from torch_geometric.utils import negative_sampling
from torch_sparse import SparseTensor

import mlflow
import mlflow.pytorch

import data
import gae
from eval import class_test


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
        "feats", type=str, help="a path to the node features (npy)"
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

edge_index = torch.LongTensor(np.load(args.edges))
features = torch.FloatTensor(np.load(args.feats))
node_classes = torch.LongTensor(np.load(args.node_classes))
full_graph = gdata.Data(
    x=features, edge_index=edge_index, node_classes=node_classes
)
full_graph.adj_t = SparseTensor(
    row=edge_index[0],
    col=edge_index[1],
    value=torch.ones(len(edge_index[1]), dtype=torch.float32),
    sparse_sizes=(len(features), len(features)),
    is_sorted=True,
)

# train test split edges
genes = torch.arange(len(node_classes))[node_classes == 0]
diseases = torch.arange(len(node_classes))[node_classes == 1]
np.logicalnp.isin(edge_index, genes)
full_graph.interclass_ei = edge_index[:]
mask = torch.rand(edge_index.shape[1]) > 0.7
full_graph.train_pos_edge_index = full_graph.edge_index[:, mask]
full_graph.val_pos_edge_index = full_graph.edge_index[:, ~mask]
full_graph.val_neg_edge_index = negative_sampling(
    full_graph.edge_index,
    num_negative_sample=full_graph.val_pos_edge_index.shape[1],
)

mlflow.set_tracking_uri("http://localhost:12345")

with mlflow.start_run():
    model = gnn.GAE(gae.Encoder(62, args.dim))
    optimizer = opt.AdamW(
        model.parameters(), args.lr, weight_decay=args.wd, amsgrad=True
    )
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20, verbose=True
    )
    model = gae.train_gae(
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

    # encode
    embeddings = gae.encode(model, full_graph, args.device)
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
        mlflow.log_metric(
            "auc_" + classes[key], classification_results[key]["roc"]
        )
        mlflow.log_metric(
            "ap_" + classes[key], classification_results[key]["ap"]
        )
