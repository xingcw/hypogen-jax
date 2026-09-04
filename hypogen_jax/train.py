"""
Train the JAX HyPoGen on a MuJoCo rollout dataset.

Mirrors train_rl_regressor.py with the hypogen approximator config:
Adam, MultiStepLR at fixed epochs, full-batch shuffling with a partial
last batch, and a parameter snapshot every save_every epochs. Evaluation
is done afterwards with the torch scripts on exported checkpoints.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from data import load_dataset
from model import HyPoGenConfig, count_params, init_params, model_losses, save_npz

jax.config.update("jax_default_matmul_precision", "highest")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout_dir", required=True)
    ap.add_argument("--domain_task", default="cheetah_run")
    ap.add_argument("--input_to_model", default="rew", choices=["rew", "dyn", "rew_dyn"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--init_seed", type=int, default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_train_epochs", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--test_fraction", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--milestones", type=int, nargs="*", default=[1500, 1800, 1900])
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--value_weight", type=float, default=0.01)
    ap.add_argument("--td_weight", type=float, default=0.01)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--weight_dim", type=int, default=128)
    ap.add_argument("--enc_dec_dim", type=int, default=256)
    ap.add_argument("--opt_block_dim", type=int, default=128)
    ap.add_argument("--num_opt_mlp_layer", type=int, default=2)
    ap.add_argument("--num_enc_dec_layer", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=8)
    ap.add_argument("--dl_din_way", default="slice")
    ap.add_argument("--dl_dw_way", default="direct")
    return ap.parse_args()


def lr_at_epoch(args, epoch):
    n_passed = sum(1 for m in args.milestones if m <= epoch)
    return args.lr * (args.gamma ** n_passed)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    data_dir = os.path.join(args.rollout_dir, args.domain_task)
    train, test, train_params, test_params = load_dataset(
        data_dir, args.domain_task, args.input_to_model, args.seed, args.test_fraction
    )
    print("train params", train_params)
    print("test params", test_params)
    n_train = train[0].shape[0]
    max_unique = int(np.unique(train[0], axis=0).shape[0])
    print(f"train rows {n_train}, unique tasks {max_unique}")

    cfg = HyPoGenConfig(
        input_dim=train[0].shape[1],
        state_dim=train[1].shape[1],
        action_dim=train[2].shape[1],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        weight_dim=args.weight_dim,
        enc_dec_dim=args.enc_dec_dim,
        num_enc_dec_layer=args.num_enc_dec_layer,
        opt_block_dim=args.opt_block_dim,
        num_opt_mlp_layer=args.num_opt_mlp_layer,
        num_layers=args.num_layers,
        dl_din_way=args.dl_din_way,
        dl_dw_way=args.dl_dw_way,
    )
    init_seed = args.seed if args.init_seed is None else args.init_seed
    params = init_params(jax.random.PRNGKey(init_seed), cfg)
    print(f"total number of params {count_params(params) / 1e6:.4f}M")

    opt = optax.inject_hyperparams(optax.adam)(learning_rate=args.lr)
    opt_state = opt.init(params)
    train_dev = [jnp.asarray(t) for t in train]

    def loss_fn(p, batch):
        loss, (aux, _) = model_losses(p, cfg, batch, max_unique, args.value_weight, args.td_weight)
        return loss, aux

    @jax.jit
    def train_step(p, s, batch):
        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, batch)
        updates, s = opt.update(grads, s, p)
        p = optax.apply_updates(p, updates)
        return p, s, aux

    @jax.jit
    def take(idx):
        return [t[idx] for t in train_dev]

    csv_path = out_dir / "train_valid.csv"
    fields = ["epoch", "lr", "loss_total", "loss_action_pred", "loss_value_pred", "loss_td", "epoch_time"]
    if not csv_path.exists():
        with open(csv_path, "w") as f:
            csv.DictWriter(f, fields).writeheader()

    rng = np.random.default_rng(args.seed)
    bs = args.batch_size
    for epoch in range(args.num_train_epochs):
        t0 = time.time()
        lr = lr_at_epoch(args, epoch)
        opt_state.hyperparams["learning_rate"] = lr
        perm = rng.permutation(n_train)
        sums = {}
        n_batches = 0
        for i in range(0, n_train, bs):
            idx = jnp.asarray(perm[i:i + bs])
            params, opt_state, aux = train_step(params, opt_state, take(idx))
            for k, v in aux.items():
                sums[k] = sums.get(k, 0.0) + v
            n_batches += 1
        row = {k: float(v) / n_batches for k, v in sums.items()}
        if not np.isfinite(row["loss_total"]):
            print(f"Epoch {epoch + 1} loss is not finite, stopping")
            break
        row.update(epoch=epoch + 1, lr=lr, epoch_time=time.time() - t0)
        with open(csv_path, "a") as f:
            csv.DictWriter(f, fields).writerow(row)
        print(f"Epoch {epoch + 1} \t Train loss {row['loss_total']:.3f} \t {row['epoch_time']:.1f}s", flush=True)
        if (epoch + 1) % args.save_every == 0:
            save_npz(ckpt_dir / f"step_{epoch + 1:08d}.npz", params)


if __name__ == "__main__":
    main()
