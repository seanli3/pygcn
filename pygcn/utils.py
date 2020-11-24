import numpy as np
import scipy.sparse as sp
import torch
import numpy as np
import scipy.sparse as sp
import os.path as osp
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid, PPI, Amazon, Reddit, Coauthor, PPI, TUDataset
from torch_geometric.utils import to_networkx
import networkx as nx
import torch
import sys
import torch_geometric.transforms as T
import os.path as osp
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid, PPI, Amazon, Reddit, Coauthor, PPI, TUDataset
from .webkb_data import WebKB
import pdb
import pickle as pkl
from scipy.sparse import coo_matrix
from torch_geometric.utils import is_undirected, to_undirected

import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def get_dataset(name, normalize_features=False, transform=None, edge_dropout=None, node_feature_dropout=None,
                dissimilar_t = 1, cuda=False, permute_masks=None, lcc=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    if name in ['Computers', 'Photo']:
        dataset = Amazon(path, name)
    elif name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(path, name, split="full")
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(path, name, split="full")
    elif name in ['Reddit']:
        dataset = Reddit(path)
    elif name.lower() in ['cornell', 'texas', 'wisconsin', 'chameleon']:
        dataset = WebKB(path, name)
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    dataset.data.y = dataset.data.y.long()
    if not is_undirected(dataset.data.edge_index):
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)

    lcc_mask = None
    if lcc:  # select largest connected component
        data_ori = dataset[0]
        data_nx = to_networkx(data_ori)
        data_nx = data_nx.to_undirected()
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = list(data_nx.nodes)

    if cuda:
        dataset.data.to('cuda')

    return dataset



def load_data(path="../data/cora/", dataset="cora"):
    if dataset=='cora':
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
    else:
        dataset = get_dataset(dataset, normalize_features=True)
        data = dataset[0]
        # pdb.set_trace()
        train_index = torch.where(data.train_mask[0])[0]

        adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                            shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
        idx_train = torch.where(data.train_mask[0])[0]
        idx_val = torch.where(data.val_mask[0])[0]
        idx_test = torch.where(data.test_mask[0])[0]
        labels = data.y
        features = data.x
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense().astype('float32')))
        features = torch.FloatTensor(features)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
