from sys import argv
import argparse

import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn
from torch_geometric.utils import train_test_split_edges

import data
import gae
from eval import class_test


torch.random.manual_seed(42)


class Classification(nn.Module):
    def __init__(self, encoder, n_classes):
        super(Classification, self).__init__()
        self.encoder = encoder
        self.lin = nn.Linear(encoder.conv3.out_channels * 5, n_classes)

    def forward(self, x, adj):
        return self.lin(self.encoder(x, adj))


best_loss = float("inf")
counter = 0


def callback(model, loss):
    global best_loss
    global counter
    if best_loss > loss:
        best_loss = loss
        counter = 0
        torch.save(model, "./best_finetuned_model.pt")
    counter += 1
    print(loss)
    if counter > 500:
        print("Stop!")
        return True
    return False


# Parameters
parser = argparse.ArgumentParser(description="Finetune pretrained model.")
parser.add_argument("lr", type=float, help="learning rate")
parser.add_argument("epochs", type=int, help="the number of epochs to train")
parser.add_argument(
    "cut", type=float, help="the threshold for cutting weak edges"
)

# Paths
parser.add_argument("model", type=str, help="a path to a serialized model")
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
target = data.labels_data(args.ids, ["FDA approved drug targets"])
full_graph.y = torch.tensor(
    target[full_graph["id"]].squeeze(), dtype=torch.long
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
    graph.train_nodes_mask = torch.rand((graph.num_nodes,)) > 0.3
    graphs.append(
        data.make_sparse(
            train_test_split_edges(graph, val_ratio=0, test_ratio=0)
        )
    )


# model preparation
mdl = torch.load(args.model)
model = Classification(mdl.encoder, 2)
loss_fn = nn.CrossEntropyLoss(
    weight=torch.tensor(
        1
        / np.unique(target, return_counts=True)[1]
        / np.sum(1 / np.unique(target, return_counts=True)[1]),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
)

# pretune linear layer
# freeze all layers but the last linear one
for param in model.parameters():
    param.requires_grad = False
for param in model.lin.parameters():
    param.requires_grad = True

optimizer = opt.AdamW(model.parameters(), 0.005)
scheduler = opt.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=20, verbose=True
)
model = gae.finetune_gae(
    model, graphs, loss_fn, optimizer, scheduler, "cpu", 200, callback
)

# finetune
# unfreeze
for param in model.parameters():
    param.requires_grad = True
optimizer = opt.AdamW(model.parameters(), args.lr)
scheduler = opt.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=20, verbose=True
)
model = gae.finetune_gae(
    model, graphs, loss_fn, optimizer, scheduler, "cpu", args.epochs, callback
)
torch.save(model, "./finetuned_model.pt")

# encode
embeddings = []
ids = []
for graph in graphs:
    embeddings.append(gae.encode(mdl, graph, "cpu"))
    ids.append(graph.id)
ids = torch.cat(ids, 0)
embeddings = np.concatenate(embeddings, 0)
embeddings = embeddings[ids.argsort().numpy()]
np.save("./embedding.npy", embeddings)

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
]

# small information leakage here!
print(class_test(embeddings, data.labels_data(args.ids, classes)))
