from lightning.pytorch.callbacks import Callback
import torch
from functools import partial
from pathlib import Path
from plotting import plot_heatmap, plot_epoch_layer_head_curves, plot_grid_heatmaps
from torch import Tensor
from jaxtyping import Float, Int
from transformer_lens import utils
from einops import reduce
import matplotlib.pyplot as plt
from torch_geometric.utils import mask_to_index
from homophily import label_informativeness, edge_homophily
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict

def get_block_index_set_from_mask(mask):
    idx = mask_to_index(mask)
    row_idx = idx.unsqueeze(1)
    col_idx = idx.unsqueeze(0)
    return (row_idx, col_idx)

def head_ablation_hook(layer, head):
    def hook_fn(z: Float[Tensor, "batch pos head_index d_head"], hook):
        z[..., head, :] = 0
        return z
    return (utils.get_act_name("z", layer=layer, layer_type="attn"), hook_fn)


class BaseAnalysisCallback(Callback, ABC):
    """Abstract base class for analysis callbacks that follow compute-plot-log pattern."""
    
    @abstractmethod
    def compute(self, pl_module):
        """Compute metrics and analysis data.
        
        Args:
            pl_module: Lightning module instance
        """
        pass
    
    @abstractmethod
    def plot(self) -> Dict[str, plt.Figure]:
        """Create plots for analysis results.
        
        Returns:
            Dict[str, plt.Figure]: Dictionary mapping plot names to figures
        """
        pass
    
    def log_plots(self, trainer, pl_module, prefix: str = ""):
        """Log all plots to trainer.
        
        Args:
            trainer: Lightning trainer instance
            pl_module: Lightning module instance
            prefix: Optional prefix for plot names
        """
        fig_dict = self.plot()
        for key, fig in fig_dict.items():
            plot_name = f"{prefix}/{key}" if prefix else key
            trainer.logger.experiment.add_figure(plot_name, fig, pl_module.current_epoch)
            plt.close(fig)
    
    def on_train_end(self, trainer, pl_module):
        """Default implementation: compute then log plots."""
        self.compute(pl_module)
        self.log_plots(trainer, pl_module)





class HeadAblationCallback(BaseAnalysisCallback):
    def __init__(self):
        super().__init__()
        self.loss_increase: Float[Tensor, "layer head"] = None

    def compute(self, pl_module):
        """Compute loss increase for each head ablation."""
        device = pl_module.device
        output = pl_module(torch.arange(pl_module.data.num_nodes), update_cache=False)
        loss_orig = output.loss
        self.loss_increase = torch.empty(pl_module.n_layer, pl_module.n_head, device=device)
        
        for layer, head in zip(range(pl_module.n_layer), range(pl_module.n_head)):
            output = pl_module(torch.arange(pl_module.data.num_nodes), update_cache=False, custom_hooks=[head_ablation_hook(layer, head)])
            self.loss_increase[layer, head] = output.loss - loss_orig

    def plot(self):
        """Create plots for head ablation results."""
        return {
            "head_ablation": plot_heatmap(
                self.loss_increase, 
                "Head Ablation Loss Increase", 
                "Head", 
                "Layer"
            )
        }


class LogTrainingAttention(BaseAnalysisCallback):
    def __init__(self, distance=2, mask=None):
        super().__init__()

        self.distance = distance
        self.mask = mask
        self.training_attention_scores = []
        self.training_edge_homophily: Float[torch.Tensor, "epoch layers heads class"] = None
        self.label_info: Float[torch.Tensor, "epoch layers heads"] = None
        self.attention_per_distance = None

    def state_dict(self):
        return {
            "distance": self.distance,
            "mask": self.mask,
            "training_attention_scores": self.training_attention_scores,
        }
    def load_state_dict(self, state_dict):
        self.distance = state_dict["distance"]
        self.mask = state_dict["mask"]
        self.training_attention_scores = state_dict["training_attention_scores"]

        
    def on_train_epoch_end(self, trainer, pl_module):
        
        scores = pl_module.stack_attention_scores()
        self.training_attention_scores.append(scores)
        
    def compute(self, pl_module):
        """Compute all attention-related metrics."""
        if isinstance(self.training_attention_scores, list):
            self.training_attention_scores: Float[Tensor, "epoch layer head n1 n2"] = torch.stack(self.training_attention_scores, dim=0)
        
        # Compute edge homophily
        edge_homophily_vmap = torch.vmap(partial(edge_homophily, y=pl_module.data.y if self.mask is None else pl_module.data.y[self.mask]))
        self.training_edge_homophily = edge_homophily_vmap(self.training_attention_scores.softmax(dim=-1))

        # Compute label informativeness
        label_informativeness_vmap = torch.vmap(partial(label_informativeness, y=pl_module.data.y if self.mask is None else pl_module.data.y[self.mask]))
        self.label_info = label_informativeness_vmap(self.training_attention_scores.softmax(dim=-1))

        # Prepare attention per distance data
        self.attention_per_distance = self._compute_attention_per_distance(pl_module)

    def _compute_attention_per_distance(self, pl_module):
        """Compute attention patterns per distance."""
        
        patterns = self.training_attention_scores.softmax(dim=-1)
        num_src, num_dst = patterns.shape[-2], patterns.shape[-1]
        masks = torch.empty(self.distance+1, num_src, num_dst, dtype=torch.bool, device=pl_module.device)
        
        # Initialize distance 0 (self-loops)
        masks[0] = torch.eye(num_src, num_dst, dtype=torch.bool, device=pl_module.device)
        
        # Track all previously covered distances to exclude them
        covered_mask = masks[0].clone()
        
        # construct mask for nodes that are reachable within the distance
        for distance in range(1, self.distance+1):
            dist_adj = pl_module.data.adj ** distance
            masked_dist_adj = dist_adj[*get_block_index_set_from_mask(self.mask)] > 0
            # Only include nodes at exactly this distance (not covered by shorter distances)
            masks[distance] = torch.logical_and(masked_dist_adj, ~covered_mask)
            # Update covered mask to include this distance
            covered_mask = torch.logical_or(covered_mask, masks[distance])

        attention_per_distance = {}
        for k in range(self.distance+1):
            p = reduce(patterns.masked_fill(~masks[k], 0), "... n1 n2 -> ...", "sum") / num_src
            attention_per_distance[k] = p.cpu()
        
        return attention_per_distance

    def plot(self):
        """Create all plots for training attention analysis."""
        fig_dict = {}
        
        # Plot edge homophily
        for i in range(self.training_edge_homophily.shape[-1]):  # num_classes
            fig_dict[f"training_edge_homophily/{i}"] = plot_epoch_layer_head_curves(
                self.training_edge_homophily[:, :, :, i], 
                "Attention Induced Edge Homophily", 
                "Epoch", 
                "Attention Induced Edge Homophily", 
                "Layer_Head"
            )
        
        fig_dict["training_edge_homophily/total"] = plot_epoch_layer_head_curves(
            reduce(self.training_edge_homophily, "epoch layer head class -> epoch layer head", reduction="sum"), 
            "Attention Induced Edge Homophily", 
            "Epoch", 
            "Attention Induced Edge Homophily", 
            "Layer_Head"
        )
        
        # Plot label informativeness
        fig_dict["training_label_informativeness"] = plot_epoch_layer_head_curves(
            self.label_info,
            title="Label Informativeness",
            xlabel="Epoch",
            ylabel="Informativeness",
            legend_title="Layer_Head"
        )
        
        # Plot attention per distance
        for k, attention_data in self.attention_per_distance.items():
            fig_dict[f"training_attention/{k}"] = plot_epoch_layer_head_curves(
                attention_data, 
                title=f"Attention Patterns (Distance {k})", 
                xlabel="Epoch", 
                ylabel="Attention", 
                legend_title="Layer_Head"
            )
        
        return fig_dict




class LogitAttribution(BaseAnalysisCallback):

    def __init__(self):
        super().__init__()
        self.head_results: Float[Tensor, "layer head node d_model"] = None
        self.logit_attrs: Float[Tensor, "layer head nodes c_true c_pred"] = None
        
        self.avg_logits_per_layer_head: Float[Tensor, "layer head c_true c_pred"] = None
        self.avg_logits_sum_all: Float[Tensor, "c_true c_pred"] = None
        
        self.true_logits_per_layer_head: Float[Tensor, "layer head c_true c_pred"] = None
        self.true_logits_sum_all: Float[Tensor, "c_true c_pred"] = None
        
        self.normed_true_logits_per_layer_head: Float[Tensor, "layer head c_true c_pred"] = None
        self.normed_true_logits_sum_all: Float[Tensor, "c_true c_pred"] = None

    def compute(self, pl_module):
        pl_module.update_cache()
        self.head_results = pl_module.stack_head_results()
        self.logit_attrs = pl_module.all_class_pair_logit_lens(self.head_results).squeeze(2)
        self.avg_logits_per_layer_head = reduce(self.logit_attrs, "layer head nodes c_true c_pred -> layer head c_true c_pred", "mean")
        self.avg_logits_sum_all = reduce(self.avg_logits_per_layer_head, "layer head c_true c_pred -> c_true c_pred", "sum")
        

        true_logits_list = []
        for class_i in range(pl_module.num_classes):
            true_logits = self.logit_attrs[..., pl_module.data.y == class_i, class_i, :].mean(dim=-2) # average over indexed nodes
            true_logits_list.append(true_logits)
        self.true_logits_per_layer_head: Float[Tensor, "layer head c_true c_pred"] = torch.stack(true_logits_list, dim=-2)
        self.true_logits_sum_all = reduce(self.true_logits_per_layer_head, "layer head c_true c_pred -> c_true c_pred", "sum")

        
        self.normed_true_logits_per_layer_head = self.true_logits_per_layer_head - self.avg_logits_per_layer_head
        self.normed_true_logits_sum_all = self.true_logits_sum_all - self.avg_logits_sum_all
    def plot(self):
        fig_dict = {
            "avg_logits_per_layer_head": plot_grid_heatmaps(
                data_tensor=self.avg_logits_per_layer_head,
                title_prefix="Average Logit Attribution per Head (All Nodes)",
                xlabel="From Class",
                ylabel="To Class"
            ),
            "avg_logits_sum_all": plot_heatmap(
                data_tensor=self.avg_logits_sum_all,
                title="Sum of Average Logit Attributions (All Nodes, All Heads/Layers)",
                xlabel="From Class",
                ylabel="To Class"
            ),
            "true_logits_per_layer_head": plot_grid_heatmaps(
                data_tensor=self.true_logits_per_layer_head,
                title_prefix="Average Logit Attribution per Head (Nodes of True Class)",
                xlabel="From Class",
                ylabel="To Class"
            ),
            "true_logits_sum_all": plot_heatmap(
                data_tensor=self.true_logits_sum_all,
                title="Sum of Average Logit Attributions (Nodes of True Class, All Heads/Layers)",
                xlabel="From Class",
                ylabel="To Class"
            ),
            "normed_true_logits_per_layer_head": plot_grid_heatmaps(
                data_tensor=self.normed_true_logits_per_layer_head,
                title_prefix="Difference: (True Class Nodes Avg Logit Attr) - (All Nodes Avg Logit Attr) per Head",
                xlabel="From Class",
                ylabel="To Class"
            ),
            "normed_true_logits_sum_all": plot_heatmap(
                data_tensor=self.normed_true_logits_sum_all,
                title="Sum of Difference: (True Class Nodes Avg Logit Attr) - (All Nodes Avg Logit Attr)",
                xlabel="From Class",
                ylabel="To Class"
            )
        }
        return fig_dict

    def log_plots(self, trainer, pl_module, prefix: str = "LogitsAttribution"):
        """Log all plots to trainer with LogitsAttribution prefix."""
        super().log_plots(trainer, pl_module, prefix)
