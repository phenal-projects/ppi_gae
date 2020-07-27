from sys import argv
import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as opt
import torch_geometric.nn as gnn
from torch_geometric.utils import train_test_split_edges

import data
import gae
from eval import class_test

best_auc = 0
counter = 0


def callback(model, auc):
    global best_auc
    global counter
    if best_auc < auc:
        best_auc = auc
        counter = 0
        torch.save(model, "./best_model.pt")
    counter += 1
    print(auc)
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
    "tissue",
    type=int,
    help="the number of a column with expression in feats file",
)
parser.add_argument("device", type=str, help="cuda or cpu")

# Paths
parser.add_argument("edges", type=str, help="a path to the edge list (tsv)")
parser.add_argument("ids", type=str, help="a path to the ids tsv")
parser.add_argument(
    "feats", type=str, help="a path to the node features (tsv)"
)

args = parser.parse_args(argv[1:])

# data loading
full_graph = data.graph_data(
    args.edges, args.feats, args.ids, cut=args.cut, sparse_tensor=False
)
expression = pd.read_csv(args.feats, sep="\t")
if 0 < args.tissue:
    full_graph = data.tissue_specific_ppi_cut(
        full_graph, full_graph.x[:, args.tissue]
    )
loader = data.cluster_data(full_graph, 1, 1, shuffle=True, verbose=True)

# make sparse tensors
graphs = []
for graph in loader:
    graph.edge_id = dict(
        list(
            zip(
                (graph.num_nodes * graph.edge_index[0] + graph.edge_index[1])
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
# encode
embeddings = []
ids = []
for graph in graphs:
    embeddings.append(gae.encode(model, graph, args.device))
    ids.append(graph.id)
ids = torch.cat(ids, 0)
embeddings = np.concatenate(embeddings, 0)
embeddings = embeddings[ids.argsort().numpy()]

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

print(class_test(embeddings, data.labels_data(args.ids, classes)))
