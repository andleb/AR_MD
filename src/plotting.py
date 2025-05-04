"""
Utility functions for plotting
"""
import os
import re
from pathlib import Path
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.colors import LogNorm
from torch.utils.data import Dataset

PLOT_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots"))
if not PLOT_DIR.is_dir():
    os.makedirs(PLOT_DIR)

from typing import List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error


def plot_joint_kde(
        x_values, y_values, show_axes: bool = True, fname: Optional[str] = None, **kwargs
):
    """
    Plot a density scatter plot (KDE) of the values. kwargs are passed to
    ax.set()

    Useful for plotting Ramachandran plot - x = phi, y = psi
    https://proteopedia.org/wiki/index.php/Ramachandran_Plots
    """
    fig, ax = plt.subplots(dpi=300)
    sns.kdeplot(x=x_values, y=y_values, levels=100, fill=True, norm=LogNorm(), ax=ax)
    if show_axes:
        ax.axvline(0, color="grey", alpha=0.5)
        ax.axhline(0, color="grey", alpha=0.5)
    ax.set(**kwargs)
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_val_dists_at_t(
        t: int,
        dset: Dataset,
        share_axes: bool = True,
        zero_center_angles: bool = False,
        fname: Optional[str] = None,
):
    select_by_attn = lambda x: x["corrupted"][torch.where(x["attn_mask"])]

    retval = []
    for i in range(len(dset)):
        vals = dset.__getitem__(i, use_t_val=t)
        assert vals["t"].item() == t, f"Unexpected values of t: {vals['t']} != {t}"
        retval.append(select_by_attn(vals))
    vals_flat = torch.vstack(retval).numpy()
    assert vals_flat.ndim == 2

    ft_names = dset.feature_names["angles"]
    n_fts = len(ft_names)
    assert vals_flat.shape[1] == n_fts

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_fts,
        sharex=share_axes,
        sharey=share_axes,
        dpi=300,
        figsize=(2.6 * n_fts, 2.5),
    )
    for i, (ax, ft_name) in enumerate(zip(axes, ft_names)):
        # Plot the values
        vals = vals_flat[:, i]
        sns.histplot(vals, ax=ax)
        if "dist" not in ft_name:
            if zero_center_angles:
                ax.axvline(np.pi, color="tab:orange")
                ax.axvline(-np.pi, color="tab:orange")
            else:
                ax.axvline(0, color="tab:orange")
                ax.axvline(2 * np.pi, color="tab:orange")
        ax.set(title=f"Timestep {t} - {ft_name}")
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_losses(
        log_fname: str,
        out_fname: Optional[str] = None,
        simple: bool = False,
        pattern: Optional[str] = None,
):
    """
    Plot the validation loss values from a log file. Spuports multiple
    validation losses if present in log file. Plots per epoch, and if multiple
    values are record for an epoch, plot the median.
    """

    def keyfunc(x: str) -> tuple:
        """
        Validation first, then train
        """
        ordering = ["test", "val", "train"]
        if "_" in x:
            x_split, x_val = x.split("_", maxsplit=1)
            x_retval = tuple([ordering.index(x_split), x_val])
        else:
            x_retval = (len(ordering), x)
        assert len(x_retval) == 2
        return x_retval

    if simple:
        assert pattern is None
        pattern = re.compile(r"_loss$")
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    fig, ax = plt.subplots(dpi=300)

    df = pd.read_csv(log_fname)
    cols = df.columns.to_list()
    cols = sorted(cols, key=keyfunc)
    for colname in df.columns:
        if "loss" not in colname:
            continue
        if pattern is not None:
            if not pattern.search(colname):
                continue
        vals = df.loc[:, ["epoch", colname]]
        vals.dropna(axis="index", how="any", inplace=True)
        sns.lineplot(x="epoch", y=colname, data=vals, ax=ax, label=colname, alpha=0.5)
    ax.legend(loc="upper right")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Loss over epochs")

    if out_fname is not None:
        fig.savefig(out_fname, bbox_inches="tight")
    return fig


def plot_consecutive_heatmap(
        vals: Union[Sequence[float], Sequence[Sequence[float]]],
        fname: Optional[str] = None,
        logstretch_vmax: float = 2e3,
        **kwargs,
):
    """
    Plot a heatmap of consecutive values.
    """
    consecutive_pairs = []

    get_pairs = lambda x: np.array(list(zip(x[:-1], x[1:])))
    # Require these more complex checks because vals may not be of the same
    # size and therefore may not be stackable
    if isinstance(vals[0], (float, int)):
        # 1-dimensional
        consecutive_pairs = get_pairs(vals)
    else:
        # 2-dimensional
        consecutive_pairs = np.vstack([get_pairs(vec) for vec in vals])
    assert consecutive_pairs.ndim == 2
    assert consecutive_pairs.shape[1] == 2

    norm = ImageNormalize(vmin=0.0, vmax=logstretch_vmax, stretch=LogStretch())

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
    density = ax.scatter_density(
        consecutive_pairs[:, 0], consecutive_pairs[:, 1], norm=norm
    )
    fig.colorbar(density, label="Points per pixel")

    ax.set(**kwargs)

    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def analyze_trajectory_prediction(
        model: torch.nn.Module,
        test_trajectories: torch.Tensor,
        initial_sequence_length: int = 100,
        prediction_length: int = 200,
        stride: int = 1,
        means: Optional[torch.Tensor] = None,
        stds: Optional[torch.Tensor] = None,
        device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """
    Analyze model predictions against actual trajectories for a batch.

    Args:
        model: Trained MDTrajectoryTransformer model
        test_trajectories: Batch of trajectories to test against
        initial_sequence_length: Number of initial frames to use for prediction
        prediction_length: Number of frames to predict forward
        stride: Prediction stride
        means: Optional means for normalization
        stds: Optional stds for normalization
        device: Device to run prediction on

    Returns:
        actual: Actual trajectory sequences
        predicted: Predicted trajectory sequences
        avg_metrics: Dictionary of averaged error metrics
        per_traj_metrics: Dictionary of per-trajectory metrics as numpy arrays
    """
    model.eval()
    model = model.to(device)

    # Get initial sequences and ground truth
    initial_sequences = test_trajectories[:, :initial_sequence_length].to(device)
    actual = test_trajectories[:, initial_sequence_length:initial_sequence_length + prediction_length].cpu()

    # Generate predictions
    with torch.no_grad():
        predicted = model.predict_trajectory(
            initial_frames=initial_sequences,
            n_steps=prediction_length,
            means=means,
            stds=stds,
            stride=stride
        )
        predicted = predicted[:, initial_sequence_length:].cpu()

    # Initialize metrics storage
    n_traj = test_trajectories.shape[0]
    mse_per_traj = np.zeros(n_traj)
    rmse_per_traj = np.zeros(n_traj)
    mae_per_traj = np.zeros(n_traj)
    rel_error_per_traj = np.zeros(n_traj)

    # Compute per-trajectory metrics
    for i in range(n_traj):
        mse_per_traj[i] = mean_squared_error(actual[i], predicted[i])
        rmse_per_traj[i] = mean_squared_error(actual[i], predicted[i], squared=False)
        mae_per_traj[i] = mean_absolute_error(actual[i], predicted[i])
        rel_error_per_traj[i] = (torch.norm(predicted[i] - actual[i]) / torch.norm(actual[i])).item()

    # Compute average metrics
    avg_metrics = {
        'mse'           : np.mean(mse_per_traj),
        'rmse'          : np.mean(rmse_per_traj),
        'mae'           : np.mean(mae_per_traj),
        'relative_error': np.mean(rel_error_per_traj)
    }

    # Store per-trajectory metrics
    per_traj_metrics = {
        'mse'           : mse_per_traj,
        'rmse'          : rmse_per_traj,
        'mae'           : mae_per_traj,
        'relative_error': rel_error_per_traj
    }

    return actual, predicted, avg_metrics, per_traj_metrics


def plot_trajectory_comparison(
        actual: torch.Tensor,
        predicted: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        title: str = "Trajectory Comparison",
        save_path: Optional[str] = None,
):
    """
    Plot comparison of actual vs predicted trajectories.

    Args:
        actual: Actual trajectory tensor [length, n_features]
        predicted: Predicted trajectory tensor [length, n_features]
        feature_names: Optional list of feature names
        title: Plot title
        save_path: Optional path to save figure
    """
    n_features = actual.shape[1]
    if feature_names is None:
        feature_names = [f"Feature {i + 1}" for i in range(n_features)]

    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
    if n_features == 1:
        axes = [axes]

    time = np.arange(len(actual))

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.plot(time, actual[:, i], label='Actual', alpha=0.7)
        ax.plot(time, predicted[:, i], label='Predicted', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel(name)
        ax.legend()
        ax.grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_prediction_error(
        actual: torch.Tensor,
        predicted: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        window_size: int = 10,
        title: str = "Prediction Error Over Time",
        save_path: Optional[str] = None,
):
    """
    Plot prediction error over time with moving average.

    Args:
        actual: Actual trajectory tensor [length, n_features]
        predicted: Predicted trajectory tensor [length, n_features]
        feature_names: Optional list of feature names
        window_size: Size of moving average window
        title: Plot title
        save_path: Optional path to save figure
    """
    n_features = actual.shape[1]
    if feature_names is None:
        feature_names = [f"Feature {i + 1}" for i in range(n_features)]

    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
    if n_features == 1:
        axes = [axes]

    time = np.arange(len(actual))
    errors = (predicted - actual).abs()

    # Compute moving average of error
    moving_avg = torch.zeros_like(errors)
    for i in range(len(errors)):
        start_idx = max(0, i - window_size + 1)
        moving_avg[i] = errors[start_idx:i + 1].mean(dim=0)

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.plot(time, errors[:, i], label='Instantaneous Error', alpha=0.3)
        ax.plot(time, moving_avg[:, i], label=f'Moving Average (window={window_size})',
                linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{name} Absolute Error')
        ax.legend()
        ax.grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
