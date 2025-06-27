# %matplotlib inline
import os
import types
from functools import partial
from pathlib import Path
import graph_tool.all as gt
import lightning as L
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import typer
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, Int
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
)
from functools import cached_property
from lightning.pytorch.loggers import TensorBoardLogger
from tensordict import TensorDict
from torch import Tensor, optim
from torch_geometric.data import Data
from torchmetrics.functional import accuracy, confusion_matrix, f1_score
from transformer_lens import (
    ActivationCache,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

from callbacks import (
    HeadAblationCallback,
    LogitAttribution,
    LogTrainingAttention,
    get_block_index_set_from_mask,
)
from plotting import (
    plot_epoch_layer_head_curves,
    plot_grid_hist,
)
from pygio import GraphDataModule, load_dataset
import numpy as np
DATASET_ROOT = "/home/lcheng/oz318/datasets/pytorch_geometric"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 32
D_HEAD = 8


class Graph(gt.Graph):
    def __init__(self, data, has_virtual_node, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.has_virtual_node = has_virtual_node
        
        self.add_vertex(data.num_nodes)
        self.add_edge_list(data.edge_index.T.cpu())        
        self.vp['y'] = self.new_vp('int')
        self.vp['y'].set_values(data.y.cpu())
        
        self.filtered_graph = self.filter_out_virtual_node()
        self.filtered_pos = gt.sfdp_layout(self.filtered_graph)
    
        pos_values = self.filtered_pos.get_2d_array()
        if self.has_virtual_node:
            pos_mean = pos_values.mean(axis=1)
            new_pos = np.concatenate((pos_values, pos_mean[:, None]), axis=1)
        else:
            new_pos = pos_values
            
        self.vp["pos"] = self.new_vp("vector<double>", new_pos.T)

    
    def filter_out_virtual_node(self):
        if self.has_virtual_node:
            vfilt = self.new_vp("bool", [True] * (self.num_vertices()-1) + [False])
            efilt = self.new_ep("bool", [True] * (self.num_edges()-self.num_vertices()) + [False] * self.num_vertices())
        else:
            vfilt = self.new_vp("bool", [True] * self.num_vertices())
            efilt = self.new_ep("bool", [True] * self.num_edges())
        g = gt.GraphView(self, vfilt=vfilt, efilt=efilt)
        return g


    def draw(self, unfiltered=False, vertex_fill_color=None, **kwargs):
        if isinstance(vertex_fill_color, torch.Tensor):
            if vertex_fill_color.dtype == torch.bool:
                vertex_fill_color = self.new_vp("bool", vertex_fill_color.cpu().numpy())
            elif vertex_fill_color.dtype == torch.float:
                vertex_fill_color = gt.prop_to_size(self.new_vp("float", vertex_fill_color), ma=1, mi=0, power=1)

        if unfiltered:
            return gt.graph_draw(self, vertex_fill_color=vertex_fill_color, pos=self.vp["pos"], bg_color=(0, 0, 0, 1), inline=True, edge_color=(0.5, 0.5, 0.5, 1.0), **kwargs)
        else:
            return gt.graph_draw(self.filtered_graph, vertex_fill_color=vertex_fill_color, pos=self.filtered_pos, bg_color=(0, 0, 0, 1), inline=True, edge_color=(0.5, 0.5, 0.5, 1.0), **kwargs)






class Experiment:
    def __init__(self, masked: bool, laplacian_pos_encoding: bool, dataset_name: str, 
                 n_layers: int, n_heads: int, max_epochs: int, add_virtual_node: bool = False,
                 device: torch.device = DEVICE, d_model: int = D_MODEL, d_head: int = D_HEAD):
        self.masked = masked
        self.laplacian_pos_encoding = laplacian_pos_encoding
        self.dataset_name = dataset_name
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_epochs = max_epochs
        self.add_virtual_node = add_virtual_node
        self.device = device
        self.d_model = d_model
        self.d_head = d_head
        


    @cached_property
    def exp_prefix(self) -> str:
        return f"{'masked' if self.masked else 'unmasked'}_{'lappe' if self.laplacian_pos_encoding else 'random'}_{self.dataset_name}_{self.n_layers}_{self.n_heads}_{'virtual' if self.add_virtual_node else 'novirtual'}"
    
    def is_experiment_completed(self) -> bool:
        checkpoint_path = Path(f"./loggings/{self.exp_prefix}/0/last.ckpt")
        return checkpoint_path.exists()
        
    @cached_property
    def data(self):
        return load_dataset(
            self.dataset_name, 
            split="full", 
            d_model=self.d_model, 
            add_virtual_node=self.add_virtual_node, 
            laplacian_pos_encoding=self.laplacian_pos_encoding
            ).to(self.device)
    
    @cached_property
    def datamodule(self):
        return GraphDataModule(self.data)

    @cached_property
    def cfg(self):
        return HookedTransformerConfig(
            d_model=self.d_model,
            d_head=self.d_head,
            n_layers=self.n_layers,
            n_ctx=self.data.num_nodes,
            n_heads=self.n_heads,
            attn_only=True,
            d_vocab=self.data.num_nodes,
            act_fn="gelu",
            attention_dir="bidirectional",
            normalization_type="LNPre",
            d_vocab_out=self.data.y.max().item() + 1,
        )

    
    @cached_property
    def model(self):
        return GraphTransformer(self.cfg, self.data, apply_neighbor_mask=self.masked)

    @cached_property
    def logger(self):
        return TensorBoardLogger(
            save_dir="./loggings",
            name=self.exp_prefix,
            version="0"
        )


    @cached_property
    def checkpoint(self):
        return ModelCheckpoint(
            dirpath=self.logger.log_dir,
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=True
        )

        
    @cached_property
    def trainer(self):
        return L.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[
                # LogTrainingAttention(mask=self.data.train_mask),
                # HeadAblationCallback(),
                # LogitAttribution(),
                self.checkpoint,
            ],
            logger=self.logger,
            log_every_n_steps=1
        )
    
    def prepare(self):
        """Prepare trainer, model, datamodule, and model checkpoint for the experiment."""
        L.seed_everything(123)
        return self.trainer, self.model, self.datamodule

    def load_model(self):
        """Load a trained model from checkpoint."""
        trainer, model, _ = self.prepare()
        ckpt = torch.load(f"{trainer.logger.log_dir}/best.ckpt", map_location=self.device)
        model.load_state_dict(ckpt['state_dict'])
        model.update_cache()
        model.to(self.device)
        model.eval()
        return model, ckpt

    def run(self):
        """Run the experiment training."""
        trainer, model, datamodule = self.prepare()
        trainer.fit(model, datamodule)

        src = Path(self.checkpoint.best_model_path)
        dst = src.parent / "best.ckpt"
        if not dst.exists():
            os.symlink(src, dst)
        return dst

    @cached_property
    def graph(self):
        return Graph(self.data, self.add_virtual_node)


    @torch.no_grad()
    def get_scores(self) -> Float[Tensor, "qk ep layer head node d_model"]:
        query_scores = self.model.query_scores(torch.stack([self.model.WE, self.model.WPos], dim=0))
        key_scores = self.model.key_scores(torch.stack([self.model.WE, self.model.WPos], dim=0))
        return torch.stack([query_scores, key_scores], dim=0)




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
    attn_scores = attn_scores.squeeze(0)  # batch size should always be 1
    attn_scores.masked_fill_(mask, -torch.inf)
    attn_scores = attn_scores.unsqueeze(0)  # recover batch size dim
    return attn_scores


def cross_entropy(data, model, logits, tokens, attention_mask=None, per_token=False):
    pred = logits.squeeze(0)
    target = data.y[tokens].squeeze(0)
    return F.cross_entropy(pred, target)


class GraphTransformer(L.LightningModule):
    def __init__(self, cfg: HookedTransformerConfig, data: Data, apply_neighbor_mask: bool, lr=1e-2, weight_decay=1e-4):
        super().__init__()
        self.cfg = cfg
        self.model = HookedTransformer(cfg)

        self.model.embed.W_E.data = data.x
        self.model.embed.W_E.requires_grad = False
        self.model.pos_embed.W_pos.data = data.pos
        self.model.pos_embed.W_pos.requires_grad = False
        self.cache_dict, self.caching_hooks, _ = self.model.get_caching_hooks(incl_bwd=False)
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
    ):
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
        # print(f"preds: {preds.argmax(dim=-1).v}, targets: {target.v}")
        self.log("train_loss", output.loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)

        df_cm = pd.DataFrame(cm.cpu().numpy(), index=range(self.num_classes), columns=range(self.num_classes))
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_cm, ax=ax, annot=True, cmap='Spectral')
        self.trainer.logger.experiment.add_figure("train_cm", fig, self.current_epoch)
        plt.close(fig)  # Close the figure to prevent it from displaying directly
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

        df_cm = pd.DataFrame(cm.cpu().numpy(), index=range(self.num_classes), columns=range(self.num_classes))
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_cm, ax=ax, annot=True, cmap='Spectral')
        self.trainer.logger.experiment.add_figure("val_cm", fig, self.current_epoch)
        plt.close(fig)  # Close the figure to prevent it from displaying directly

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

        df_cm = pd.DataFrame(cm.cpu().numpy(), index=range(self.num_classes), columns=range(self.num_classes))
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_cm, ax=ax, annot=True, cmap='Spectral')
        self.trainer.logger.experiment.add_figure("test_cm", fig, self.current_epoch)
        plt.close(fig)  # Close the figure to prevent it from displaying directly

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
        patterns = attention_patterns.mean(dim=-1).cpu()  # shape: (epochs, layers, heads)
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
    def query_side(self) -> Float[Tensor, "layer head d_model d_head"]:
        return self.QK.U * self.QK.S[..., None, :].sqrt()

    @property
    def key_side(self) -> Float[Tensor, "layer head d_head d_model"]:
        return self.QK.S[..., :, None].sqrt() * self.QK.Vh.transpose(-1, -2)

    def query_scores(self, embeddings: Float[Tensor, "... node d_model"]) -> Float[Tensor, "... layer head node d_head"]:
        cache = self.get_cache()
        embeddings = cache.apply_ln_to_stack(embeddings)
        query_scores = einsum(embeddings, self.query_side, "... node d_model, layer head d_model d_head -> ... layer head node d_head")
        return query_scores

    def key_scores(self, embeddings: Float[Tensor, "... node d_model"]) -> Float[Tensor, "... layer head node d_head"]:
        cache = self.get_cache()
        embeddings = cache.apply_ln_to_stack(embeddings)
        key_scores = einsum(embeddings, self.key_side, "... node d_model, layer head d_head d_model -> ... layer head node d_head")
        return key_scores





app = typer.Typer()


@app.command()
def run_experiment(
    masked: bool = typer.Option(True, "--masked/--unmasked", help="Whether to apply neighbor masking"),
    laplacian_pos_encoding: bool = typer.Option(True, "--lappe/--random", help="Use Laplacian positional encoding"),
    dataset_name: str = typer.Option("cora", help="Dataset name (cora or citeseer)"),
    n_layers: int = typer.Option(2, help="Number of layers"),
    n_heads: int = typer.Option(2, help="Number of attention heads"),
    add_virtual_node: bool = typer.Option(False, "--virtual/--no-virtual", help="Add virtual node to the graph"),
    max_epochs: int = typer.Option(100, help="Maximum number of training epochs"),
    force: bool = typer.Option(False, "--force", help="Force rerun experiment even if already completed"),
):
    """Run a graph transformer experiment with the specified configuration."""
    config = Experiment(
        masked=masked,
        laplacian_pos_encoding=laplacian_pos_encoding,
        dataset_name=dataset_name,
        n_layers=n_layers,
        n_heads=n_heads,
        max_epochs=max_epochs,
        add_virtual_node=add_virtual_node
    )
    
    if not force and config.is_experiment_completed():
        typer.echo(f"Experiment already completed: {config.exp_prefix}")
        raise typer.Exit()

    typer.echo(f"Starting experiment: {config.exp_prefix}")
    config.run()


if __name__ == "__main__":
    app()
    # run(masked=True, laplacian_pos_encoding=True, dataset_name="cora", n_layers=2, n_heads=2, add_virtual_node=True)
    # run(masked=True, laplacian_pos_encoding=False, dataset_name="cora", n_layers=2, n_heads=2, add_virtual_node=True)
    # run(masked=True, laplacian_pos_encoding=True, dataset_name="cora", n_layers=2, n_heads=2, add_virtual_node=False)
    # run(masked=True, laplacian_pos_encoding=False, dataset_name="cora", n_layers=2, n_heads=2, add_virtual_node=False)
