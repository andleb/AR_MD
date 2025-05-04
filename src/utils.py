"""
Misc shared utility functions
"""
import os
import glob
import hashlib
import logging
from typing import *

from typing import Optional

import requests

import numpy as np
import matplotlib.pyplot as plt
import lightning as pl

from pytorch_lightning.callbacks import Callback


def is_huggingface_hub_id(s: str) -> bool:
    """
    Return True if s looks like a repo ID
    >>> is_huggingface_hub_id("wukevin/foldingdiff_cath")
    True
    >>> is_huggingface_hub_id("wukevin/foldingdiff_cath_lol")
    False
    """
    r = requests.get(f"https://huggingface.co/{s}")
    return r.status_code == 200


def extract(a, t, x_shape):
    """
    Return the t-th item in a for each item in t
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def num_to_groups(num: int, divisor: int) -> List[int]:
    """
    Generates a list of ints of value at most divisor that sums to

    >>> num_to_groups(18, 16)
    [16, 2]
    >>> num_to_groups(33, 8)
    [8, 8, 8, 8, 1]
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    assert sum(arr) == num
    return arr


def seq_to_groups(seq:Sequence[Any], divisor:int) -> List[Sequence[Any]]:
    """
    Generates a list of items of at most <divisor> items
    >>> seq_to_groups([1,2,3,4,5,6,7,8,9], 3)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> seq_to_groups([1,2,3,4,5,6,7,8,9], 4)
    [[1, 2, 3, 4], [5, 6, 7, 8], [9]]
    """
    return [seq[i:i+divisor] for i in range(0, len(seq), divisor)]


def tolerant_comparison_check(values, cmp: Literal[">=", "<="], v):
    """
    Compares values in a way that is tolerant of numerical precision
    >>> tolerant_comparison_check(-3.1415927410125732, ">=", -np.pi)
    True
    """
    if cmp == ">=":  # v is a lower bound
        minval = np.nanmin(values)
        diff = minval - v
        if np.isclose(diff, 0, atol=1e-5):
            return True  # Passes
        return diff > 0
    elif cmp == "<=":
        maxval = np.nanmax(values)
        diff = maxval - v
        if np.isclose(diff, 0, atol=1e-5):
            return True
        return diff < 0
    else:
        raise ValueError(f"Illegal comparator: {cmp}")


def modulo_with_wrapped_range(
    vals, range_min: float = -np.pi, range_max: float = np.pi
):
    """
    Modulo with wrapped range -- capable of handing a range with a negative min

    >>> modulo_with_wrapped_range(3, -2, 2)
    -1
    """
    assert range_min <= 0.0
    assert range_min < range_max

    # Modulo after we shift values
    top_end = range_max - range_min
    # Shift the values to be in the range [0, top_end)
    vals_shifted = vals - range_min
    # Perform modulo
    vals_shifted_mod = vals_shifted % top_end
    # Shift back down
    retval = vals_shifted_mod + range_min

    # Checks
    # print("Mod return", vals, " --> ", retval)
    # if isinstance(retval, torch.Tensor):
    #     notnan_idx = ~torch.isnan(retval)
    #     assert torch.all(retval[notnan_idx] >= range_min)
    #     assert torch.all(retval[notnan_idx] < range_max)
    # else:
    #     assert (
    #         np.nanmin(retval) >= range_min
    #     ), f"Illegal value: {np.nanmin(retval)} < {range_min}"
    #     assert (
    #         np.nanmax(retval) <= range_max
    #     ), f"Illegal value: {np.nanmax(retval)} > {range_max}"
    return retval


def update_dict_nonnull(d: Dict[str, Any], vals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a dictionary with values from another dictionary.
    >>> update_dict_nonnull({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
    {'a': 1, 'b': 3, 'c': 4}
    """
    for k, v in vals.items():
        if k in d:
            if d[k] != v and v is not None:
                logging.info(f"Replacing key {k} original value {d[k]} with {v}")
                d[k] = v
        else:
            d[k] = v
    return d


def md5_all_py_files(dirname: str) -> str:
    """Create a single md5 sum for all given files"""
    # https://stackoverflow.com/questions/36099331/how-to-grab-all-files-in-a-folder-and-get-their-md5-hash-in-python
    fnames = glob.glob(os.path.join(dirname, "*.py"))
    hash_md5 = hashlib.md5()
    for fname in sorted(fnames):
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(2**20), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()



class LossTracker(Callback):
    def __init__(self):

        super().__init__()

        self.training_losses = []
        self.training_steps = []
        self.current_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "training_losses": self.training_losses,
            "training_steps": self.training_steps,
            "current_epoch": self.current_epoch
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.training_losses = state_dict["training_losses"]
        self.training_steps = state_dict["training_steps"]
        self.current_epoch = state_dict["current_epoch"]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict):
            loss = outputs['loss']
        else:
            loss = outputs
        self.training_losses.append(loss.item())
        self.training_steps.append(trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        self.current_epoch += 1

    def plot_losses(
        self,
        moving_avg_window: int = 100,
        title: str = "Training Loss",
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6),
        ylim: Optional[tuple] = None
    ):
        if not self.training_losses:
            print("No losses collected yet!")
            return

        losses = np.array(self.training_losses)
        steps = np.array(self.training_steps)

        if len(losses) >= moving_avg_window:
            moving_avg = np.convolve(losses, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
            moving_avg_steps = steps[moving_avg_window-1:]
        else:
            moving_avg = None
            print(f"Warning: Not enough data points for moving average with window {moving_avg_window}")

        plt.figure(figsize=figsize)
        plt.plot(steps, losses, alpha=0.3, label='Training Loss')

        if moving_avg is not None:
            plt.plot(moving_avg_steps, moving_avg,
                    linewidth=2, label=f'Moving Average (window={moving_avg_window})')

        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title(title)
        if ylim:
            plt.ylim(ylim)
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        plt.show()



def plot_training_losses(
    trainer,
    moving_avg_window: int = 100,
    title: str = "Training Loss",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot training losses with optional moving average.

    Args:
        trainer: PyTorch Lightning trainer after training
        moving_avg_window: Window size for moving average smoothing
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    # Extract losses from trainer
    losses = [batch['loss'].item() for batch in trainer.fit_loop.epoch_loop._results.batch_indices.values()]
    epochs = np.arange(len(losses))

    # Calculate moving average
    moving_avg = np.convolve(losses, np.ones(moving_avg_window)/moving_avg_window, mode='valid')

    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(epochs, losses, alpha=0.3, label='Training Loss')
    plt.plot(epochs[moving_avg_window-1:], moving_avg,
             linewidth=2, label=f'Moving Average (window={moving_avg_window})')

    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()



if __name__ == "__main__":
    import doctest

    doctest.testmod()
