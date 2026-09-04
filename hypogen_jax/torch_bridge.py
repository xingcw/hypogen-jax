"""
Torch-side helpers for the JAX port. Run from the repo root with the
main (torch) environment.

Subcommands:
  dump-sd    save a torch rl_net.pt state_dict as an npz of float32 arrays
  reference  dump inputs, outputs, losses and gradients of the torch model
             on a fixed batch, for the JAX equivalence test
  export     write JAX npz checkpoints as rl_net.pt files in a
             results_approximator-style directory that batch_eval_regressor
             can read
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def dump_sd(args):
    import torch

    sd = torch.load(args.pt, map_location="cpu")
    np.savez(args.out, **{k: v.detach().float().numpy() for k, v in sd.items()})
    print(f"saved {len(sd)} tensors to {args.out}")


def _build_torch_model(workdir, device):
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from train_rl_regressor import make_approximator

    cfg = OmegaConf.load(Path(workdir) / "cfg.yaml")
    input_dim = {"rew": 1, "dyn": 1, "rew_dyn": 2}[cfg.input_to_model]
    approx = make_approximator(input_dim, args_state_dim(cfg), args_action_dim(cfg), cfg.approximator, device=device)
    return approx, cfg


def args_state_dim(cfg):
    return {"cheetah_run": 17, "walker_walk": 24, "finger_spin": 9}[cfg.domain_task]


def args_action_dim(cfg):
    return {"cheetah_run": 6, "walker_walk": 6, "finger_spin": 2}[cfg.domain_task]


def reference(args):
    import torch
    import torch.nn.functional as F

    from hypogen_jax.data import load_dataset

    torch.manual_seed(0)
    approx, cfg = _build_torch_model(args.workdir, "cpu")
    if args.pt:
        approx.rl_net.load_state_dict(torch.load(args.pt, map_location="cpu"))
    approx.train(True)
    approx.rl_net.train()

    data_dir = os.path.join(args.rollout_dir, cfg.domain_task)
    train, _, train_params, _ = load_dataset(
        data_dir, cfg.domain_task, cfg.input_to_model, cfg.seed, cfg.test_fraction
    )
    checks = {f"check_sum_{i}": np.float64(t.astype(np.float64).sum()) for i, t in enumerate(train)}
    checks["check_rows"] = np.int64(train[0].shape[0])

    b = args.batch_size
    n = train[0].shape[0]
    idx = np.arange(b) if args.batch_mode == "first" else np.linspace(0, n - 1, b).astype(int)
    batch = [torch.tensor(t[idx]) for t in train]
    input_param, state, action, next_state, reward, discount, value = batch

    task_emb, pred_action, pred_value = approx.rl_net(input_param, state, action, train=True)
    task_emb.retain_grad()
    loss_action = F.mse_loss(pred_action, action)
    loss_value = F.mse_loss(pred_value, value)
    with torch.no_grad():
        next_action = approx.rl_net.predict_action(task_emb, next_state)
    target_q = approx.rl_net.predict_q_value(task_emb, next_state, next_action)
    target_q = reward + discount * target_q
    loss_td = F.mse_loss(value, target_q)
    loss = loss_action + approx.value_weight * loss_value + approx.td_weight * loss_td

    approx.rl_net_optimizer.zero_grad()
    loss.backward()

    out = {f"in_{i}": t.numpy() for i, t in enumerate(batch)}
    out.update(
        z=task_emb.detach().numpy(),
        pred_action=pred_action.detach().numpy(),
        pred_q=pred_value.detach().numpy(),
        next_action=next_action.numpy(),
        target_q=target_q.detach().numpy(),
        grad_z=task_emb.grad.numpy(),
        loss_total=np.float32(loss.item()),
        loss_action_pred=np.float32(loss_action.item()),
        loss_value_pred=np.float32(approx.value_weight * loss_value.item()),
        loss_td=np.float32(approx.td_weight * loss_td.item()),
    )
    out.update(checks)
    for name, p in approx.rl_net.named_parameters():
        if p.grad is not None:
            out["grad." + name] = p.grad.detach().numpy()
    np.savez(args.out, **out)
    np.savez(args.out_sd, **{k: v.detach().float().numpy() for k, v in approx.rl_net.state_dict().items()})
    print(f"loss {loss.item():.6f} action {loss_action.item():.6f}; wrote {args.out} and {args.out_sd}")


def export(args):
    import torch

    out_dir = Path(args.out_dir)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(args.template_workdir) / "cfg.yaml", out_dir / "cfg.yaml")
    npz_paths = sorted(Path(args.ckpt_dir).glob("step_*.npz"))
    if args.steps:
        npz_paths = [p for p in npz_paths if int(re.search(r"step_(\d+)", p.name).group(1)) in args.steps]
    for npz in npz_paths:
        step = int(re.search(r"step_(\d+)", npz.name).group(1))
        with np.load(npz) as f:
            sd = {k: torch.from_numpy(np.array(f[k])) for k in f.files}
        d = models_dir / f"step_{step:08d}"
        d.mkdir(exist_ok=True)
        torch.save(sd, d / "rl_net.pt")
        if args.best_step is not None and step == args.best_step:
            bd = models_dir / "step_best_reward"
            bd.mkdir(exist_ok=True)
            torch.save(sd, bd / "rl_net.pt")
    print(f"exported {len(npz_paths)} checkpoints to {models_dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("dump-sd")
    s.add_argument("--pt", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=dump_sd)

    s = sub.add_parser("reference")
    s.add_argument("--workdir", required=True, help="torch run dir containing cfg.yaml")
    s.add_argument("--pt", default=None, help="checkpoint to load; random init if omitted")
    s.add_argument("--rollout_dir", required=True)
    s.add_argument("--batch_size", type=int, default=512)
    s.add_argument("--batch_mode", default="strided", choices=["first", "strided"])
    s.add_argument("--out", required=True)
    s.add_argument("--out_sd", required=True)
    s.set_defaults(fn=reference)

    s = sub.add_parser("export")
    s.add_argument("--ckpt_dir", required=True)
    s.add_argument("--out_dir", required=True)
    s.add_argument("--template_workdir", required=True)
    s.add_argument("--steps", type=int, nargs="*", default=None)
    s.add_argument("--best_step", type=int, default=None)
    s.set_defaults(fn=export)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
