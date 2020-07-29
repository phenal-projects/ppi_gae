import pandas as pd

import torch
from torch_geometric.data import Data, ClusterData, ClusterLoader
from torch_geometric.utils import is_undirected
from torch_geometric.transforms import ToSparseTensor
from torch_sparse import SparseTensor
import numpy as np


def graph_data(
    edge_list_path,
    node_features_path,
    protein_ids_path,
    protein_id_col_node="Gene",
    protein_id_col_prot="ensembl.gene",
    sparse_tensor=True,
    cut=0,
):
    """Creates a data object from the given tsv files.

    Parameters
    ----------
    edge_list_path : str
        Path to edge list file. The first two columns -- edges, the rest are
        edge attributes.
    node_features_path : str
        Path to a file with node features.
    protein_ids_path : str
        Protein ids filepath. File should contain
    protein_id_col_node : str, optional
        Column with ids in the node features file, by default "Gene"
    protein_id_col_prot : str, optional
        Column with ids in the protein_ids_file, by default "ensembl.gene"
    sparse_tensor : bool, optional
        If true, a sparse tensor will be constructed instead of edge index and
        edge_weight, by default True
    cut : int, optional
        Edges with values below the cut will be dropped, by default 0

    Returns
    -------
    torch_geometric.data.Data
        Data file with a graph
    """
    a = pd.read_csv(edge_list_path).values
    edge_attr = a[:, 2:] / 1000.0

    # cut the edges
    cut_mask = edge_attr[:, -1] > cut
    edge_ind = torch.tensor(a[:, :2][cut_mask], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr[cut_mask], dtype=torch.float32)

    # force undirected
    if not is_undirected(edge_ind):
        edge_ind = torch.cat([edge_ind, edge_ind[:, [1, 0]]], 0)
        edge_attr = torch.cat([edge_attr, edge_attr], 0)

    # features
    protein_ids = pd.read_csv(protein_ids_path, sep="\t")[
        ["id", protein_id_col_prot]
    ]
    x = pd.read_csv(node_features_path, sep="\t")
    feature_columns = x.drop(protein_id_col_node, 1).columns
    x = pd.merge(
        protein_ids,
        x,
        how="left",
        left_on=protein_id_col_prot,
        right_on=protein_id_col_node,
    ).sort_values("id")[feature_columns]
    x.fillna(x.mean(), inplace=True)
    x = torch.tensor(((x - x.mean()) / x.std()).values, dtype=torch.float32)
    data = Data(x, edge_ind.T, edge_attr, id=torch.arange(x.shape[0]))

    if sparse_tensor:
        tsp = ToSparseTensor(False)
        data = tsp(data)

    return data


def cluster_data(data, num_clusters, batch_size, shuffle=True, verbose=True):
    """Prepares clusters for batching

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data object.
    num_clusters : int
        The number of clusters to chop the input graph into.
    batch_size : int
        The number of clusters in each batch
    shuffle : bool, optional
        If true, the ClusterLoader will shuffle clusters, by default True
    verbose : bool, optional
        If true, prints clusters info, by default True

    Returns
    -------
    torch_geometric.data.ClusterLoader
        A loader for training
    """
    clusters = ClusterData(data, num_clusters, recursive=True, save_dir=None)
    if verbose:
        for cluster in clusters:
            print(cluster)
    return ClusterLoader(clusters, batch_size=batch_size)


def labels_data(protein_data_path, columns):
    """Prepares numpy labels data for classification test

    Parameters
    ----------
    protein_data_path : str
        Path to a file with labels
    columns : list of str
        Columns in the dataset to use

    Returns
    -------
    numpy.ndarray
        binary matrix with classes' labels
    """
    labels = pd.read_csv(protein_data_path, sep="\t").fillna(0)
    return labels[columns].astype(int).values


def make_sparse(data):
    """This method transforms train_pos_edge_index in the data
    object into a sparse tensor.

    Parameters
    ----------
    data : torch_geometric.data.Data
        A graph data object

    Returns
    ----------
    torch_geometric.data.Data
        Transformed graph object
    """
    assert data.train_pos_edge_index is not None

    (row, col), N = data.train_pos_edge_index, data.num_nodes
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = [data.edge_id[(row[i] * N + col[i]).item()].item() for i in perm]

    data.adj_t = SparseTensor(
        row=col,
        col=row,
        value=torch.tensor(value, dtype=torch.float32),
        sparse_sizes=(N, N),
        is_sorted=True,
    )

    # Pre-process some important attributes.
    data.adj_t.storage.rowptr()
    data.adj_t.storage.csr2csc()

    return data


def tissue_specific_ppi(data, expr_level):
    """Constructs tissue-specific PPI network via reweighting the edges.
    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data object.
    expr_level : torch.Tensor
        tensor with expression levels
    Returns
    -------
    torch_geometric.data.Data
        New reweighted graph
    """
    row, col = data.edge_index
    data.edge_attr = (
        data.edge_attr.squeeze() * expr_level[row] * expr_level[col]
    )
    data.edge_attr = data.edge_attr / data.edge_attr.max()
    data.edge_attr = data.edge_attr.reshape(-1, 1)
    return data


def tissue_specific_ppi_cut(data, expr_level, threshold=1):
    """Constructs tissue-specific PPI network via removing proteins.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data object.
    expr_level : torch.Tensor
        tensor with expression levels
    threshold : float, optional
        cutting threshold

    Returns
    -------
    torch_geometric.data.Data
        New graph
    """
    row, col = data.edge_index
    data.new_id = data.id[expr_level > threshold]
    mask = torch.tensor(
        np.logical_and(np.isin(row, data.new_id), np.isin(col, data.new_id))
    )
    data.edge_index = data.edge_index[:, mask]
    data.edge_attr = data.edge_attr[mask]
    data.expr_mask = mask
    return data
