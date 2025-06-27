import torch
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from jaxtyping import Float
from typing import Union


def plot_grid_hist(scores: Float[torch.Tensor, "hue *grid node"], title: str, xlabel, stat="density", columns=None, row_names="layer", col_names="head", axes=None) -> Union[plt.Figure, None]:
    # Extract grid dimensions similar to plot_grid_heatmaps
    _, rows, cols, _ = scores.shape
    df_list = []
    ax_name = lambda i, j: f'{row_names}_{i}_{col_names}_{j}'
    for i in range(rows):
        for j in range(cols):
            df_temp = pd.DataFrame(scores[:, i, j].detach().cpu().numpy().T, columns=columns)
            df_temp['source'] = ax_name(i, j)
            df_list.append(df_temp)
    df = pd.concat(df_list, ignore_index=True)
    
    # Create figure and axes if not provided
    if axes is None:
        fig, axes = plt.subplots(figsize=(cols * 5, rows * 4), nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=False)
        created_fig = True
    else:
        fig = None
        created_fig = False
        # Ensure axes is 2D array
        if hasattr(axes, 'shape') and len(axes.shape) == 1:
            axes = axes.reshape(rows, cols)
        elif not hasattr(axes, '__getitem__'):
            # Single axis provided for grid - this would be an error
            raise ValueError(f"Expected {rows}x{cols} grid of axes, but got single axis")
    
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j] if hasattr(axes[i], '__getitem__') else axes[i]
            # Filter data for this specific layer-head combination
            subset_data = df[df['source'] == ax_name(i, j)]
            sns.histplot(subset_data.drop(columns=['source']), ax=ax, stat=stat)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(stat)
            ax.set_title(ax_name(i, j))
    
    if created_fig:
        fig.suptitle(title)
        fig.tight_layout()
        return fig
    return None

def plot_heatmap(data_tensor: torch.Tensor, title: str, xlabel: str, ylabel: str, ax=None) -> Union[plt.Figure, None]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    else:
        fig = None
        created_fig = False
        
    sns.heatmap(data_tensor.cpu().detach(), cbar=True, ax=ax, center=0, annot=True, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if created_fig:
        return fig
    return None


def get_outliers_mask(scores, n=3, dim=-2):
    upper = (scores.mean(dim=dim) + n * scores.std(dim=dim)).unsqueeze(dim)
    lower = (scores.mean(dim=dim) - n * scores.std(dim=dim)).unsqueeze(dim)
    return (scores < lower) | (scores > upper)

# def plot_scores_hist(scores, qk, k):
    # scores = scores[qk, ..., k]
    # return plot_grid_hist(scores, title="Query Scores", xlabel="Score", stat="density", columns=["WE", "WPos"], row_names="layer", col_names="head")



def plot_grid_heatmaps(data_tensor: Float[Tensor, "*grid c1 c2"], title_prefix: str, xlabel: str, ylabel: str, axes=None) -> Union[plt.Figure, None]:
    # rows, cols = data_tensor.shape[2:]
    rows, cols, _, _ = data_tensor.shape
    
    # Create figure and axes if not provided
    if axes is None:
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 7, rows * 6),  # Increased figure size for titles
                                 squeeze=False)
        created_fig = True
    else:
        fig = None
        created_fig = False
        # Ensure axes is 2D array
        if hasattr(axes, 'shape') and len(axes.shape) == 1:
            axes = axes.reshape(rows, cols)
        elif not hasattr(axes, '__getitem__'):
            # Single axis provided for grid - this would be an error
            raise ValueError(f"Expected {rows}x{cols} grid of axes, but got single axis")
    
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j] if hasattr(axes[i], '__getitem__') else axes[i]
            sns.heatmap(data_tensor[i, j, :, :].cpu().detach(), cbar=True, ax=ax, center=0, annot=True, cmap="viridis")
            ax.set_title(f"Layer {i}, Head {j}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
    
    if created_fig:
        fig.suptitle(title_prefix, fontsize=16, y=1.02) # Add super title for the grid
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
        return fig
    return None


def plot_epoch_layer_head_curves(
    data: Union[pd.DataFrame, Float[torch.Tensor, "epochs layers heads"]], 
    title: str, 
    xlabel: str, 
    ylabel: str, 
    legend_title: str,
    ax=None
) -> Union[plt.Figure, None]:
    """Plot line curves showing evolution of metrics over epochs for different layer-head combinations.
    
    Args:
        data: Either a melted DataFrame with columns ['epoch', 'layer_head', 'attention'] 
              or a tensor with shape (epochs, layers, heads) which will be converted to DataFrame
        title: Plot title
        xlabel: X-axis label  
        ylabel: Y-axis label
        legend_title: Legend title
        ax: Optional matplotlib axis to plot on
        
    Returns:
        plt.Figure or None: The created figure if ax was None, otherwise None
    """
    # Convert tensor to DataFrame if needed
    if isinstance(data, torch.Tensor):
        df_long = create_melted_dataframe_from_tensor(data)
    else:
        df_long = data
    
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
        created_fig = True
    else:
        fig = None
        created_fig = False
        
    sns.lineplot(
        data=df_long,
        x="epoch",
        y="attention",
        hue="layer_head",
        marker="o",
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
    
    if created_fig:
        plt.tight_layout()
        return fig
    return None


def create_melted_dataframe_from_tensor(patterns_tensor: Float[torch.Tensor, "epochs layers heads"]):
    """Create a melted DataFrame from a tensor for plotting time series data.
    
    Args:
        patterns_tensor: Tensor with shape (epochs, layers, heads)
        
    Returns:
        pd.DataFrame: Melted DataFrame with columns 'epoch', 'layer_head', 'attention'
    """
    epochs, n_layers, n_heads = patterns_tensor.shape
    df = pd.DataFrame(
        patterns_tensor.reshape(epochs, -1).cpu().detach().numpy(),
        columns=[f"layer{l}_head{h}" for l in range(n_layers) for h in range(n_heads)]
    )
    df["epoch"] = np.arange(epochs)
    return df.melt(id_vars="epoch", var_name="layer_head", value_name="attention")

