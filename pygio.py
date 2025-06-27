import torch
import lightning as L
from scipy.sparse.linalg import eigsh
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import mask_to_index, dense_to_sparse
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
)

import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data
import numpy as np


import graph_tool.all as gt

PYG_DATASET_ROOT = "/home/lcheng/oz318/datasets/pytorch_geometric"

def laplacian(data, normalization="sym", dtype=None):
    edge_index = data.edge_index
    edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
    edge_index, edge_weight = get_laplacian(edge_index=edge_index, edge_weight=edge_weight, normalization=normalization, num_nodes=data.num_nodes, dtype=dtype)
    L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)
    return L.todense()

def lap_pe(data, k, normalization="sym", return_eigvals=False, **kwargs):
    L = laplacian(data, normalization=normalization)
    eig_vals, eig_vecs = eigsh(  # type: ignore
        L,
        k=k,
        which='SA',
        return_eigenvectors=True,
        **kwargs,
    )
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = torch.from_numpy(eig_vecs)

    for i in range(pe.shape[1]):
        max_abs_idx = torch.argmax(torch.abs(pe[:, i]))
        if pe[max_abs_idx, i] < 0:
            pe[:, i] *= -1
    if return_eigvals:
        return pe, eig_vals[:k]
    return pe

class StandardizeNodeFeatures(T.BaseTransform):
    def __init__(self, eps=1e-6, keys=["x", "pos"]):
        super().__init__()
        self.eps = eps
        self.keys = keys

    def forward(self, data):
        for key in self.keys:
            mean = data[key].mean(dim=0)
            variance = data[key].var(dim=0)
            data[key] = (data[key] - mean) / (variance + self.eps).sqrt()
        return data


class VirtualNode(T.VirtualNode):
    def forward(self, data):
        data = super().forward(data)
        mean_features = data.x[:-1].mean(dim=0)
        data.x[-1] = mean_features
        return data


class AddNumClasses(T.BaseTransform):
    def __init__(self, y="y", attr="num_classes"):
        super().__init__()
        self.y = y
        self.attr = attr
    def __call__(self, data: Data):
        if hasattr(data, self.y) and data.y is not None:
            data[self.attr] = int(torch.unique(data[self.y]).numel())
        else:
            print(f"Warning: cannot compute num_classes, 'y' missing on Data.")
        return data
    def __repr__(self) -> str:
        return 'AddNumClasses()'
    
class SVDFeatureReduction(T.BaseTransform):
    r"""Dimensionality reduction of node features via Singular Value
    Decomposition (SVD) (functional name: :obj:`svd_feature_reduction`).

    Args:
        out_channels (int): The dimensionality of node features after
            reduction.
    """
    def __init__(self, out_channels: int):
        self.out_channels = out_channels

    def forward(self, data: Data) -> Data:
        assert data.x is not None
        data.orig = data.x.clone()  # Store original features
        U, S, V = torch.linalg.svd(data.x)
        # data.x = U[:, :self.out_channels] @ torch.diag(S[:self.out_channels])
        data.x = U[:, :self.out_channels]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'


# Add GraphDataModule class definition here
class GraphDataModule(L.LightningDataModule):
    def __init__(self, data: Data):
        super().__init__()
        self.data = data

    def train_dataloader(self):
        return mask_to_index(self.data.train_mask).unsqueeze(0)

    def val_dataloader(self):
        return mask_to_index(self.data.val_mask).unsqueeze(0)

    def test_dataloader(self):
        return mask_to_index(self.data.test_mask).unsqueeze(0)
    
    def predict_dataloader(self):
        return torch.arange(self.data.num_nodes).unsqueeze(0)

class OrthogonalPositionalEncoding(T.BaseTransform):
    def __init__(self, k, attr_name="pos"):
        super().__init__()
        self.k = k
        self.attr_name = attr_name
    def forward(self, data):
        data[self.attr_name] = torch.nn.init.orthogonal_(torch.empty(data.num_nodes, self.k))
        return data
    



class AddLaplacianEigenvectorPE(T.BaseTransform):
    r"""Adds the Laplacian eigenvector positional encoding from the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    # Number of nodes from which to use sparse eigenvector computation:
    SPARSE_THRESHOLD: int = 100

    def __init__(
        self,
        k: int,
        attr_name = 'laplacian_eigenvector_pe',
        is_undirected: bool = False,
        **kwargs,
    ) -> None:
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        assert data.num_nodes is not None
        pe, eigvals = lap_pe(data, k=self.k, normalization='sym', return_eigvals=True, **self.kwargs)
        data[self.attr_name] = pe
        data["eigvals"] = torch.from_numpy(eigvals)
        return data


def load_dataset(dataset_name, split="public", num_train_per_class=20, num_val=500, num_test=1000, d_model=128, add_virtual_node=False, self_loops=True, laplacian_pos_encoding=True):
    """Load a dataset (Cora, Citeseer, or PubMed) with specified transformations."""
    # Map dataset names to Planetoid dataset names
    dataset_map = {
        "cora": "Cora",
        "citeseer": "Citeseer", 
        "pubmed": "PubMed"
    }
    
    if dataset_name.lower() not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(dataset_map.keys())}")
    
    # Build transforms
    transforms = []
    if add_virtual_node:
        transforms.append(VirtualNode())
    if self_loops:
        transforms.append(T.AddSelfLoops())

    transforms.append(T.ToUndirected())
        
    if laplacian_pos_encoding:
        transforms.append(AddLaplacianEigenvectorPE(k=d_model, attr_name="pos"))
    else:
        transforms.append(OrthogonalPositionalEncoding(k=d_model, attr_name="pos"))
        
        
    transforms.extend([
        SVDFeatureReduction(out_channels=d_model),
        T.ToDense(),
        AddNumClasses(),
        # StandardizeNodeFeatures(eps=1e-6, keys=["x", "pos"])
    ])
    
    transforms = T.Compose(transforms)
    
    # Load dataset
    planetoid_name = dataset_map[dataset_name.lower()]
    datasets = Planetoid(
        root=PYG_DATASET_ROOT, 
        name=planetoid_name, 
        transform=transforms, 
        split=split, 
        num_train_per_class=num_train_per_class, 
        num_val=num_val, 
        num_test=num_test
    )
    
    data = datasets[0]  # single graph instance
    data.edge_index, data.edge_weight = dense_to_sparse(data.adj.to_dense())
    return data
