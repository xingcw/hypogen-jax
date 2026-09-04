"""
Numpy loader for the HyPoGen rollout datasets.

Reproduces the split and loading order of utils.dataset.RLSolutionDataset
(including its float rounding of the train fraction) without torch.
"""

import os
import re
from pathlib import Path

import numpy as np
from numpy.random import default_rng


KEYS = ["state", "action", "next_state", "reward", "discount", "value"]


def _split(params, seed, test_fraction, train_fraction):
    rng = default_rng(seed)
    rng.shuffle(params)
    test_size = int(test_fraction * len(params))
    train_fraction = train_fraction if train_fraction else 1.0 - test_fraction
    train_size = min(int(train_fraction * len(params)), len(params) - test_size)
    return params[test_size:test_size + train_size], params[:test_size]


def get_split(data_dir, domain_task, input_to_model, seed, test_fraction, train_fraction=None):
    datadir = Path(data_dir)
    paths = sorted(datadir.glob(f"**/{domain_task}*_seed_*.npy"))
    if input_to_model == "rew":
        names = [os.path.basename(p) for p in paths]
        params = sorted(set(re.findall(r"\-(.*?)\_", p)[0] for p in names))
        params = [x for x in params if "-0.0" not in x]
    elif input_to_model == "dyn":
        params = sorted(set(re.findall(r"\_dyn_(.*?)\__", str(p))[0] for p in paths))
    elif input_to_model == "rew_dyn":
        params = sorted(set(re.findall(r"linear-(.*?)__", str(p))[0] for p in paths))
    else:
        raise ValueError(input_to_model)
    return _split(params, seed, test_fraction, train_fraction)


def load_stage(data_dir, domain_task, input_to_model, params):
    """Load and flatten every rollout file of the given task parameters, in order."""
    datadir = Path(data_dir)
    data = {}
    for r in params:
        for p in sorted(datadir.glob(f"**/{domain_task}*_seed_*{r}_*.npy")):
            d = np.load(str(p), allow_pickle=True).item()
            for k, v in d.items():
                n_ep, n_steps = v.shape[0], v.shape[1]
                data.setdefault(k, []).append(v.reshape(n_ep * n_steps, -1))
    data = {k: np.concatenate(v, axis=0) for k, v in data.items()}
    data["reward_dynamics_param"] = np.concatenate(
        [data["reward_param"], data["dynamics_param"]], axis=-1
    )
    in_key = {"rew": "reward_param", "dyn": "dynamics_param", "rew_dyn": "reward_dynamics_param"}
    cols = [data[in_key[input_to_model]]] + [data[k] for k in KEYS]
    return [np.ascontiguousarray(c, dtype=np.float32) for c in cols]


def load_dataset(data_dir, domain_task, input_to_model, seed, test_fraction, train_fraction=None):
    train_params, test_params = get_split(
        data_dir, domain_task, input_to_model, seed, test_fraction, train_fraction
    )
    train = load_stage(data_dir, domain_task, input_to_model, train_params)
    test = load_stage(data_dir, domain_task, input_to_model, test_params)
    return train, test, train_params, test_params
