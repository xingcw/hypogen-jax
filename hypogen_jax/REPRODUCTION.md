# HyPoGen JAX on TPU: reproduction report

Reproduction of the cheetah_run reward-variation experiment with the JAX port,
run on a TPU v4-8. Written 2026-09-05.

## Result

cheetah_run, seed 123, `input_to_model=rew`, `test_fraction=0.8`.
Protocol: 32 held-out speeds x 10 episodes per checkpoint, best checkpoint
selected by mean return (`batch_eval_regressor.py`).

| run                    | best epoch | mean return     | training time              |
| ---------------------- | ---------- | --------------- | -------------------------- |
| torch (official, GPU)  | 860        | 744.95 +/- 171.76 | about 4.3 h to epoch 860 |
| JAX, GPU               | 460        | 836.0 +/- 87.1  | about 30 min to epoch 460  |
| JAX, TPU v4 (one chip) | 660        | 857.87 +/- 90.22 | 16 min to epoch 660; 49.8 min for all 2000 |
| paper                  | -          | 819.23 +/- 81.62 | -                          |

The `+/-` column is `mean_std` as `batch_eval_regressor.py` defines it (root of
the mean per-speed episode variance). Spread across the 32 speeds is wider:
155.28 at the best checkpoint.

Next best checkpoints: epoch 1140 (844.71), 1440 (843.34), 1120 (841.61).

### Caveats

- Single seed, single task. The full grid is 5 seeds x 3 tasks x 3 experiments.
- "Best epoch" is the max over 99 evaluated checkpoints, which biases the number
  upward. The metric is noisy along the trajectory (epoch 420 gives 636.8,
  epoch 620 gives 805.3, epoch 1240 gives 692.4), so the gap between 857.87 and
  the 836.0 of the GPU run is probably within noise. Both runs used the same
  selection protocol, so they are comparable to each other.
- 99 of 100 checkpoints evaluated. Epoch 1060 failed with mjWARN_BADCTRL
  ("Physics state is invalid"). That checkpoint is healthy: 0 non-finite values
  across its 728 tensors, max abs parameter 5.189 against 5.15 and 5.21 for its
  neighbours at 1040 and 1080. The failure is a diverging rollout, not a bad
  checkpoint.

## Timing

Measured on one v4 chip, batch 512, 137 steps per epoch:

- XLA compile: about 158 s (two programs: the scan body and the partial last batch)
- steady state: 1.40 s/epoch
- whole run: 49.8 min for 2000 epochs

2.8x faster per epoch than the GPU run, using one of the four chips.

## Environment

Training needs only `hypogen_jax/pyproject.toml` (`jax[tpu]`); it resolves to
jax 0.11.1 and libtpu 0.0.46.1.

Evaluation needs torch and dm_control, installed separately into the repo-root
`.venv` (python 3.9). Only the subset the eval path imports is required;
`hypnettorch`, `cvxpy`, `stable-baselines3`, `mani-skill2`, opencv and seaborn
are not. Two constraints matter:

- `qpth` must be `>=0.0.18`. `learn2learn` pulls `qpth==0.0.15`, whose setup.py
  declares an invalid requirement (`numpy>=1<2`) and fails to build.
- `numpy` must be `<2`. gym 0.22 does not support numpy 2.

```bash
uv venv --python 3.9 .venv
uv pip install --python .venv/bin/python torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python \
  "dm-control==1.0.15" "mujoco==3.0.0" "gym==0.22.0" "pyparsing==3.0.0" \
  "qpth>=0.0.18" "numpy<2" dm-env dm-tree hydra-core omegaconf scipy tqdm \
  joblib natsort einops matplotlib tensorboard gitpython learn2learn \
  hydra-joblib-launcher \
  "contextual-control-suite @ git+https://github.com/SAIC-MONTREAL/contextual-control-suite.git"
```

A TPU VM has no libEGL, so the torch entrypoints fall back to `MUJOCO_GL=osmesa`.

## Reproducing

### 1. Train

```bash
cd hypogen_jax
.venv/bin/python train.py \
  --rollout_dir ../rollout_data/rollout_data_grid_v4_rew \
  --domain_task cheetah_run --input_to_model rew --seed 123 \
  --out_dir <run_dir>
```

Everything else defaults to `cfgs/mujoco_rew_exp.yaml`: 2000 epochs, batch 512,
`test_fraction` 0.8, lr 5e-4, milestones 1500/1800/1900, gamma 0.5,
value/td weight 0.01, snapshot every 20 epochs. 100 snapshots is about 12 GB,
so keep `out_dir` off a small disk.

### 2. Build a cfg.yaml

`torch_bridge.py export` and `batch_eval_regressor.py` both read a `cfg.yaml`
that a torch run would have written. Compose it from the repo configs rather
than hand-writing it:

```python
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
with initialize_config_dir(config_dir="<repo>/cfgs", version_base=None):
    cfg = compose(config_name="mujoco_rew_exp",
                  overrides=["domain_task=cheetah_run", "seed=123",
                             "device=cpu", "input_to_model=rew"])
OmegaConf.save(cfg, "cfg.yaml")
```

This resolves to `dl_din_way=slice`, `dl_dw_way=direct`, `hyper_type=hypogen`,
`num_layers=8`, matching the JAX defaults. Note that the top-level
`dl_din_way: direct` in `cfgs/mujoco_rew_exp.yaml` is dead config: the
approximator block takes the literal `slice` from `cfgs/approximator/hypogen.yaml`
and nothing interpolates the top-level key.

### 3. Export to torch

`batch_eval_regressor.py` derives method, input_to_model and seed from the
directory names, so the layout must be
`<root>/<exp_name>/<method>/<input_to_model>/<domain_task>/seed_<n>/models/step_*`.

```bash
.venv/bin/python hypogen_jax/torch_bridge.py export \
  --ckpt_dir <run_dir>/ckpt \
  --out_dir <root>/rew_exp/hypogen/rew/cheetah_run/seed_123 \
  --template_workdir <dir containing cfg.yaml>
```

### 4. Evaluate

`--step_to_load` takes one checkpoint per invocation, so sweeping all 100 means
running it 100 times; parallelise across steps rather than inside one run
(`--n_jobs 1` per process, `OMP_NUM_THREADS=1`). One checkpoint over 32 speeds
x 10 episodes takes about 115 s, so the full sweep is roughly 10 min at 32
workers. Must be run from the repo root: the test-parameter lists are read from
the relative path `./rollout_data/...`.

```bash
OMP_NUM_THREADS=1 .venv/bin/python batch_eval_regressor.py \
  --approximator_rootdir <root> \
  --domain_task_list cheetah_run --exp_name_list rew_exp --seeds 123 \
  --n_episodes 10 --n_jobs 1 --device cpu --step_to_load <step> \
  --output_dir <out>/step_<step>
```

## Changes made for TPU, and why

`--matmul_precision`, default `high`. The port hardcoded `highest`, which on GPU
is plain float32 and costs nothing, but on TPU is float32 emulated with six
bfloat16 passes. `high` is three passes. `test_equivalence.py` still asserts
against `highest`.

The epoch runs under `lax.scan` instead of a Python loop that uploaded one index
slice per step. Same host RNG stream, same batches, partial last batch still run.

The row-to-task map is computed once on the host instead of by a `jnp.unique`
sort inside every step. Every part of the hypernetwork is row-independent along
the task axis, so the dataset-wide unique set gives the same result as the
per-batch one.

`--per_task_apply` (off by default, untested on hardware) evaluates the target
nets task-major and selects each row's task afterwards, replacing a batch of B
matrix-vector products and about 230 MB/step of weight gathers with real
matmuls, at a factor U more target-net FLOPs.

`--drop_last` (off by default) skips the partial last batch so XLA compiles one
program instead of two. It changes the trajectory.

### Equivalence checks

Run on the CPU backend against the pre-change code:

| change                | difference from the original                       |
| --------------------- | -------------------------------------------------- |
| precomputed task map  | loss bit-identical; gradients within 4.7e-8 rel L2  |
| `--per_task_apply`    | loss 9.6e-8, gradients 1.1e-7 rel L2                |
| `lax.scan` epoch      | final params within 6.2e-13 rel L2 over 3 epochs, identical per-epoch losses |

Covered both batches containing every task and batches missing one.

## Why only one chip

Data parallelism over the four chips is not worth it for this model. Measured
from the lowered HLO at U=7 unique tasks:

```
per-step dot FLOPs at B=512: 5.398 GFLOP
per-step dot FLOPs at B=128: 4.486 GFLOP
  batch-independent (hypernetwork on U rows): 4.182 GFLOP = 77%
  batch-dependent   (target nets on B rows):  1.216 GFLOP = 23%
```

77% of the work is on the U=7 task axis and does not shrink with batch size, so
sharding the batch four ways cuts per-chip work only 1.20x while adding a 127 MB
gradient all-reduce every step against a 10.2 ms step. The right use of four
chips is four independent runs (seeds or tasks), which is communication-free.

Effective throughput is about 0.53 TFLOP/s, low for a v4. That is a shape
problem, not a chip-count problem: the hypernetwork batch dimension is 7 and the
target-net output dimensions are 1 and 6, against a 128x128 MXU.
`--per_task_apply` is the lever for that and has not been measured yet.

A side effect of the 77% figure: larger batches are nearly free per step, so
raising batch size would cut epoch time by roughly 2.4x at 2048. It changes the
optimisation and would need the learning rate retuned, so it was not used here.

## Not done

- `--per_task_apply` and `--matmul_precision highest` were never benchmarked on
  the chip.
- Only seed 123 on cheetah_run/rew. The remaining 44 runs of the grid.
- No evaluation during training. The torch trainer rolls out the environment
  every 20 epochs and keeps a `best_reward` snapshot; the JAX trainer only logs
  training losses, so best-epoch selection is entirely post-hoc.
