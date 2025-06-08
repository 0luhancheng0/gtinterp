import torch
import lightning as L
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import mask_to_index, subgraph, dense_to_sparse
import os 

from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np


import graph_tool.all as gt

PYG_DATASET_ROOT = "/home/lcheng/oz318/datasets/pytorch_geometric"

class StandardizeNodeFeatures(T.BaseTransform):
    def __init__(self, eps=1e-6, keys=["x"]):
        super().__init__()
        self.eps = eps
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            mean = data[key].mean(dim=0)
            variance = data[key].var(dim=0)
            data[key] = (data[key] - mean) / (variance + self.eps).sqrt()
        return data



def pyg_to_gt(data, directed=False):
    """
    Convert PyTorch Geometric data object to graph-tool Graph.

    graph = pyg_to_gt(data)
    vertex_color = graph.new_vp('float', qk[0, 0, :, 7].AB.squeeze().detach().cpu())
    gt.graph_draw(graph, vertex_fill_color=vertex_color, pos=graph.vp['pos'])
    """
    graph = gt.Graph(directed=directed)
    graph.add_vertex(data.num_nodes)
    graph.add_edge_list(data.edge_index.T.cpu())
    graph.vp['y'] = graph.new_vertex_property('int')
    graph.vp['y'].set_values(data.y.cpu())
    graph.vp['pos'] = gt.sfdp_layout(graph)
    return graph

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
        if data.x.size(-1) > self.out_channels:
            U, S, _ = torch.linalg.svd(data.x)
            data.x = torch.mm(U[:, :self.out_channels],
                              torch.diag(S[:self.out_channels]))

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


def load_dataset(dataset_name, split="public", num_train_per_class=20, num_val=500, num_test=1000, d_model=128, add_virtual_nodes=False, self_loops=True, undirected=True):
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
    if add_virtual_nodes:
        transforms.append(VirtualNode())
    if self_loops:
        transforms.append(T.AddSelfLoops())
    if undirected:
        transforms.append(T.ToUndirected())
        
    transforms.extend([
        SVDFeatureReduction(out_channels=d_model),
        T.AddLaplacianEigenvectorPE(k=d_model),
        T.ToDense(),
        AddNumClasses(),
        StandardizeNodeFeatures(eps=1e-6, keys=["x", "laplacian_eigenvector_pe"])
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
