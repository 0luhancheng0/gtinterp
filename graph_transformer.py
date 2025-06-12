%matplotlib inline
from torch_geometric.utils import to_undirected, degree, get_ppr, get_laplacian
from lightning.pytorch.utilities import grad_norm
from pathlib import Path
import pandas as pd
import graph_tool.all as gt
from pygio import pyg_to_gt
from einops import einsum, reduce, rearrange, repeat
from torch_geometric.transforms import VirtualNode
from plotting import plot_heatmap, plot_grid_heatmaps, plot_epoch_layer_head_curves
import plotly.express as px
import matplotlib.pyplot as plt
from callbacks import LogTrainingAttention, LogitAttribution, HeadAblationCallback, get_block_index_set_from_mask
from pygio import GraphDataModule, load_dataset

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import circuitsvis as cv
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    train,
    utils,
    ActivationCache,
    SVDInterpreter
)

from torch import Tensor
import torch
from tensordict import TensorDict
import torch.nn.functional as F
import types
from torch_geometric.utils import mask_to_index, homophily, to_cugraph, to_networkx, subgraph, k_hop_subgraph, dense_to_sparse, index_to_mask
from torchmetrics.functional import accuracy, confusion_matrix, f1_score

from jaxtyping import Float, Int
from transformer_lens.hook_points import HookPoint

from torch import optim
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from torch_geometric.data import Data
from functools import partial


import einops
import numpy as np
DATASET_ROOT = "/home/lcheng/oz318/datasets/pytorch_geometric"



def attention_mask_hook_factory(
    adj: Float[torch.Tensor, "nodes nodes"],
    idx: Int[torch.Tensor, "pos"],
    attn_scores: Float[torch.Tensor, "batch n_head pos pos"],
    hook: HookPoint,
) -> Float[torch.Tensor, "batch n_head pos pos"]:

    row_idx = idx.unsqueeze(1).to(attn_scores.device)
    col_idx = idx.unsqueeze(0).to(attn_scores.device)
    mask = (~adj.bool())[row_idx, col_idx]
    mask = repeat(mask, "query_pos key_pos -> n_head query_pos key_pos", n_head=attn_scores.shape[1])
    attn_scores = attn_scores.squeeze(0) # batch size should always be 1
    attn_scores.masked_fill_(mask, -torch.inf)
    attn_scores = attn_scores.unsqueeze(0) # recover batch size dim
    return attn_scores


def cross_entropy(data, model, logits, tokens, attention_mask=None, per_token=False):
    pred = logits.squeeze(0)
    target = data.y[tokens].squeeze(0)
    return F.cross_entropy(pred, target)




class GraphTransformer(L.LightningModule):
    def __init__(self, cfg: HookedTransformerConfig, data: Data, apply_neighbor_mask: bool, lr=1e-2, weight_decay=1e-4):
        super().__init__()
        # self.save_hyperparameters()
        self.cfg = cfg
        self.model = HookedTransformer(cfg)

        self.model.embed.W_E.data = data.x
        self.model.embed.W_E.requires_grad = False
        # if random_orthogonal_pos_init:
        #     self.model.pos_embed.W_pos.data = torch.nn.init.orthogonal(torch.empty(data.num_nodes, cfg.d_model))
        # else:
        self.model.pos_embed.W_pos.data = data.pos
        self.model.pos_embed.W_pos.requires_grad = False
        self.cache_dict, self.caching_hooks, _ = self.model.get_caching_hooks(incl_bwd=False)
        # self.register_module("cache_dict", self.cache_dict)
        self.data = data
        self.apply_neighbor_mask = apply_neighbor_mask

        loss_fn = types.MethodType(partial(cross_entropy, self.data), self.model)
        self.model.loss_fn = loss_fn
        self.lr = lr
        self.attention_mask_hook_lambda = lambda idx: [
            (utils.get_act_name("attn_scores", layer=layer, layer_type="attn"), partial(attention_mask_hook_factory, self.data.adj, idx))
            for layer in range(self.cfg.n_layers)
        ]

        self.num_classes = self.data.y.max().item() + 1
        self.weight_decay = weight_decay
        # self.register_parameter("cache_dict", self.cache_dict)
    @property
    def n_layer(self):
        return self.cfg.n_layers
    @property
    def n_head(self):
        return self.cfg.n_heads
    def stack_attention_patterns(self):
        cache = self.get_cache()
        return torch.cat([cache[utils.get_act_name("pattern", layer=layer, layer_type="attn")] for layer in range(self.cfg.n_layers)])
    def stack_attention_scores(self):
        cache = self.get_cache()
        return torch.cat([cache[utils.get_act_name("attn_scores", layer=layer, layer_type="attn")] for layer in range(self.cfg.n_layers)])
    @torch.no_grad()
    def stack_head_results(self) -> Float[torch.Tensor, "layer head node d_model"]:
        cache = self.get_cache()
        head_results = cache.stack_head_results()
        head_results = rearrange(head_results, "(layer head) ... -> layer head ...", layer=self.cfg.n_layers, head=self.cfg.n_heads)
        return head_results
    @property
    def WE(self):
        return self.model.embed.W_E.data
    @property
    def WPos(self):
        return self.model.pos_embed.W_pos.data
    @property
    def WU(self):
        return self.model.unembed.W_U.data
    @property
    def QK(self):
        return self.model.QK
    @property
    def OV(self):
        return self.model.OV
    @property
    def QK_full(self):
        return (self.WE + self.WPos) @ self.model.QK @ (self.WE + self.WPos).T
    @property
    def OV_full(self):
        return (self.WE + self.WPos) @ self.model.OV @ self.WU

    def forward(self, idx, update_cache=False, custom_hooks=[], **kwargs):
        fwd_hooks = []
        if update_cache:
            fwd_hooks += self.caching_hooks
        if self.apply_neighbor_mask:
            fwd_hooks += self.attention_mask_hook_lambda(idx)
        fwd_hooks += custom_hooks
        output = self.model.run_with_hooks(
            idx,
            tokens=idx,
            return_type="both",
            fwd_hooks=fwd_hooks,
            **kwargs,
        )
        return output

    def training_step(
        self, batch: TensorDict, batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        idx = batch
        output = self(idx, update_cache=True)
        preds = output.logits.squeeze()
        target = self.data.y[idx]
        acc = accuracy(
            preds,
            target,
            task="multiclass",
            num_classes=self.num_classes,
        )
        f1 = f1_score(
            preds,
            target,
            task="multiclass",
            num_classes=self.num_classes,
            average="macro"
        )
        cm = confusion_matrix(
            preds,
            target,
            task="multiclass",
            num_classes=self.num_classes,
        )

        self.log("train_loss", output.loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)

        df_cm = pd.DataFrame(cm.cpu().numpy(), index = range(self.num_classes), columns=range(self.num_classes))
        fig, ax = plt.subplots(figsize = (10,7))
        sns.heatmap(df_cm, ax=ax, annot=True, cmap='Spectral')
        self.trainer.logger.experiment.add_figure("train_cm", fig, self.current_epoch)
        plt.close(fig) # Close the figure to prevent it from displaying directly
        return output.loss

        
    def validation_step(
        self, batch: TensorDict, batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        idx = batch
        output = self(idx)
        preds = output.logits.squeeze()
        target = self.data.y[idx]
        acc = accuracy(
            preds,
            target,
            task="multiclass",
            num_classes=self.num_classes,
        )
        f1 = f1_score(
            preds,
            target,
            task="multiclass",
            num_classes=self.num_classes,
            average="macro"
        )
        cm = confusion_matrix(
            preds,
            target,
            task="multiclass",
            num_classes=self.num_classes,
        )


        self.log("val_loss", output.loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

        df_cm = pd.DataFrame(cm.cpu().numpy(), index = range(self.num_classes), columns=range(self.num_classes))
        fig, ax = plt.subplots(figsize = (10,7))
        sns.heatmap(df_cm, ax=ax, annot=True, cmap='Spectral')
        self.trainer.logger.experiment.add_figure("val_cm", fig, self.current_epoch)
        plt.close(fig) # Close the figure to prevent it from displaying directly

    def test_step(
        self, batch: TensorDict, batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        idx = batch
        output = self(idx)
        preds = output.logits.squeeze()
        target = self.data.y[idx]
        acc = accuracy(
            preds,
            target,
            task="multiclass",
            num_classes=self.num_classes,
        )
        f1 = f1_score(
            preds,
            target,
            task="multiclass",
            num_classes=self.num_classes,
            average="macro"
        )
        cm = confusion_matrix(
            preds,
            target,
            task="multiclass",
            num_classes=self.num_classes,
        )
        self.log("test_loss", output.loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)

        df_cm = pd.DataFrame(cm.cpu().numpy(), index = range(self.num_classes), columns=range(self.num_classes))
        fig, ax = plt.subplots(figsize = (10,7))
        sns.heatmap(df_cm, ax=ax, annot=True, cmap='Spectral')
        self.trainer.logger.experiment.add_figure("test_cm", fig, self.current_epoch)
        plt.close(fig) # Close the figure to prevent it from displaying directly

        return output.loss


    def get_cache(self):
        return ActivationCache(self.cache_dict, self.model)

    @torch.no_grad()
    def update_cache(self):
        return self(torch.arange(self.data.num_nodes), update_cache=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=10,
            factor=0.5,
            min_lr=1e-4,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to monitor for scheduler
            },
        }

    def get_attention_patterns(self):
        cache = self.get_cache()
        patterns = torch.stack([
            cache[utils.get_act_name(f"pattern{i}attn")].squeeze() 
            for i in range(self.cfg.n_layers)
        ])
        if patterns.dim() == 3:
            patterns = repeat(patterns, "... -> layer ...", layer=1)
        return patterns
    def all_true_mask(self):
        return torch.ones(self.data.num_nodes, dtype=torch.bool).to(self.device)
    def plot_attention_patterns(self, layer_index, head_index, mask=None):
        pattern = self.get_attention_patterns()
        pattern = pattern[layer_index, head_index, *get_block_index_set_from_mask(mask if mask is not None else self.all_true_mask())]
        return sns.heatmap(pattern.cpu().numpy(), cbar=False)

    # need to check
    def plot_training_attention_per_layer_head(self, attention_patterns: Float[torch.Tensor, "epochs layers heads edges"] = None):
        # Prepare the attention data for plotting
        patterns = attention_patterns.mean(dim=-1).cpu()  # shape: (epochs, layers, heads)

        # Plot
        return plot_epoch_layer_head_curves(
            patterns,
            title="Attention Values over Epochs",
            xlabel="Epoch",
            ylabel="Attention",
            legend_title="Layer_Head"
        )



    @torch.no_grad()
    def all_class_pair_logit_lens(self, resid: Float[torch.Tensor, "layer head *batch_pos d_model"]) -> Float[torch.Tensor, "layer head *batch_pos c1 c2"]:
        logit_directions = self.all_class_pair_logit_directions()
        cache = self.get_cache()
        scaled_residual_stack = cache.apply_ln_to_stack(
            resid,
            layer=-1,
        )
        return einsum(scaled_residual_stack, logit_directions, "... d_model, d_model c1 c2 -> ... c1 c2")

    @property
    def all_class_pair_indices(self):
        allpairidx = torch.cartesian_prod(torch.arange(self.num_classes), torch.arange(self.num_classes))
        return allpairidx

    @torch.no_grad()
    def all_class_pair_logit_directions(self) -> Float[Tensor, "... c1 c2"]:
        WU = self.model.unembed.W_U
        directions = WU[:, self.all_class_pair_indices[:, 0]] - WU[:, self.all_class_pair_indices[:, 1]]
        directions = rearrange(directions, "d_model (c1 c2) -> d_model c1 c2", c1=self.num_classes, c2=self.num_classes)
        return directions
    @property
    def query_side(self) -> Float[Tensor, "layer head d_model"]:
        return self.QK.U * self.QK.S[..., None, :].sqrt()
    @property
    def key_side(self) -> Float[Tensor, "layer head d_model"]:
        return self.QK.S[..., :, None].sqrt() * self.QK.Vh.transpose(-1, -2)
    
def get_samples_maximise_loss(model, data, num_samples=100):
    output = model(torch.arange(data.num_nodes).to(device))
    per_sample_loss = F.cross_entropy(output.logits.squeeze(), data.y, reduction="none")
    idx = per_sample_loss.topk(num_samples, dim=-1)
    y_true = data.y[idx.indices]
    y_pred = output.logits.squeeze(0)[idx.indices].argmax(dim=-1)
    return output, y_true, y_pred, idx



d_model = 32
max_epochs = 50
d_head = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare(masked, random_orthogonal_pos_init, dataset_name, n_layers, n_heads):
    exp_prefix = f"{'masked' if masked else 'unmasked'}_{'random' if random_orthogonal_pos_init else 'lappe'}_{dataset_name}_{n_layers}_{n_heads}"
    
    L.seed_everything(123)
    data = load_dataset(dataset_name, split="full", d_model=d_model, random_orthogonal_pos_init=random_orthogonal_pos_init).to(device)
    datamodule = GraphDataModule(data)
    cfg = HookedTransformerConfig(
        d_model=d_model,
        d_head=d_head,
        n_layers=n_layers,
        n_ctx=data.num_nodes,
        n_heads=n_heads,
        attn_only=True,
        d_vocab=data.num_nodes,
        act_fn="gelu",
        attention_dir="bidirectional",
        d_vocab_out=data.y.max().item() + 1,
    )


    logger = TensorBoardLogger(
        save_dir="./loggings",
        name=exp_prefix,
        version="0"
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            LogTrainingAttention(mask=data.train_mask),
            LogitAttribution(),
            HeadAblationCallback(),
            ModelCheckpoint(
                save_last=True,
                save_on_train_epoch_end=False
            )
        ],
        logger=logger,
    )

    model = GraphTransformer(cfg, data, apply_neighbor_mask=masked)
    return trainer, model, datamodule

def load_model(masked, random_orthogonal_pos_init, dataset_name, n_layers, n_heads):
    trainer, model, _ = prepare(masked, random_orthogonal_pos_init, dataset_name, n_layers, n_heads)
    ckpt = torch.load(f"{trainer.logger.log_dir}/checkpoints/last.ckpt", map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.update_cache()
    return model

def run(masked, random_orthogonal_pos_init, dataset_name, n_layers, n_heads):
    trainer, model, datamodule = prepare(masked, random_orthogonal_pos_init, dataset_name, n_layers, n_heads)
    trainer.fit(model, datamodule)
    return trainer

def run_all():
    for dataset_name in ["cora", "citeseer"]:
        for n_layers, n_heads in [(1, 1), (2, 1), (1, 2), (2, 2)]:
            for masked in [True, False]:
                for random_orthogonal_pos_init in [True, False]:
                    run(masked=masked, random_orthogonal_pos_init=random_orthogonal_pos_init, dataset_name=dataset_name, n_layers=n_layers, n_heads=n_heads)

# n_heads = 2
# n_layers = 2
# dataset_name = "cora"
# masked = True
# random_orthogonal_pos_init = True
# run(masked=masked, random_orthogonal_pos_init=random_orthogonal_pos_init, dataset_name=dataset_name, n_layers=n_layers, n_heads=n_heads)

# trainer, model, datamodule = prepare(masked=True, random_orthogonal_pos_init=True, dataset_name="cora", n_layers=2, n_heads=2)




lap = load_model(masked=True, random_orthogonal_pos_init=False, dataset_name="cora", n_layers=2, n_heads=2)
ortho = load_model(masked=True, random_orthogonal_pos_init=True, dataset_name="cora", n_layers=2, n_heads=2)



def plot_grid_hist(scores: Float[torch.Tensor, "layer head node hue"], title: str, xlabel, ylabel, columns=None):
    df_list = []
    n_layers, n_heads = scores.shape[:2]
    for layer in range(n_layers):
        for head in range(n_heads):
            df_temp = pd.DataFrame(scores[layer, head].detach().cpu().numpy(), columns=columns)
            df_temp['source'] = f'layer_{layer}_head_{head}'
            df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    fig, axes = plt.subplots(figsize=(10, 6), nrows=n_layers, ncols= n_heads, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        sns.histplot(df, ax=ax)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f"Layer {i//n_layers}, Head {i%n_heads}")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


model = lap
singular_direction_idx = 2



torch.kl_div((model.WE @ model.query_side)[..., 0].log(), (model.WPos @ model.query_side)[..., 0])

query_scores = torch.cat([
    model.WE @ model.query_side[..., singular_direction_idx, None], 
    model.WPos @ model.query_side[..., singular_direction_idx, None],
], dim=-1)

plot_grid_hist(query_scores, title="How well does WE align with U*sqrt(S)", xlabel="Scores on singular vectors", ylabel="Count", columns=["WE", "WPos"])

key_scores = torch.cat([
    model.WE @ model.key_side.transpose(-1, -2)[..., singular_direction_idx, None], 
    model.WPos @ model.key_side.transpose(-1, -2)[..., singular_direction_idx, None],
], dim=-1)

plot_grid_hist(key_scores, title="How well does WE align with sqrt(S)*V.T", xlabel="Scores on singular vectors", ylabel="Count", columns=["WE", "WPos"])