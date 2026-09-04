"""
L0/L1 check: compare the JAX model against a torch reference dump made by
torch_bridge.py reference. Reports data-loader parity, forward outputs,
losses and gradients.
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from data import load_dataset
from model import HyPoGenConfig, count_params, load_npz, model_losses

jax.config.update("jax_default_matmul_precision", "highest")


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--state_dict", required=True)
    ap.add_argument("--rollout_dir", required=True)
    ap.add_argument("--domain_task", default="cheetah_run")
    ap.add_argument("--input_to_model", default="rew")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--test_fraction", type=float, default=0.8)
    ap.add_argument("--tol_fwd", type=float, default=1e-4)
    ap.add_argument("--tol_grad", type=float, default=1e-3)
    ap.add_argument("--x64", action="store_true", help="also compute a float64 reference in JAX")
    args = ap.parse_args()
    if args.x64:
        jax.config.update("jax_enable_x64", True)

    ref = dict(np.load(args.reference))
    params = load_npz(args.state_dict)
    batch = [jnp.asarray(ref[f"in_{i}"]) for i in range(7)]
    cfg = HyPoGenConfig(
        input_dim=batch[0].shape[1], state_dim=batch[1].shape[1], action_dim=batch[2].shape[1]
    )
    print(f"params {count_params(params)}")

    train, _, _, _ = load_dataset(
        os.path.join(args.rollout_dir, args.domain_task),
        args.domain_task, args.input_to_model, args.seed, args.test_fraction,
    )
    ok = True
    for i, t in enumerate(train):
        d = abs(float(t.astype(np.float64).sum()) - float(ref[f"check_sum_{i}"]))
        ok &= d < 1e-3 * (abs(float(ref[f"check_sum_{i}"])) + 1.0)
    ok &= int(ref["check_rows"]) == train[0].shape[0]
    print(f"[data] rows {train[0].shape[0]} vs {int(ref['check_rows'])}, checksums match: {ok}")

    max_unique = int(np.unique(np.asarray(batch[0]), axis=0).shape[0])

    def loss_fn(p):
        loss, (aux, dbg) = model_losses(p, cfg, batch, max_unique, 0.01, 0.01)
        return loss, (aux, dbg)

    (loss, (aux, dbg)), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(params)

    print("[forward]")
    worst_fwd = 0.0
    for k in ["z", "pred_action", "pred_q", "next_action", "target_q"]:
        e = rel_err(dbg[k], ref[k])
        worst_fwd = max(worst_fwd, e)
        print(f"  {k:12s} rel_err {e:.2e}  shape {tuple(np.asarray(dbg[k]).shape)}")
    for k in ["loss_total", "loss_action_pred", "loss_value_pred", "loss_td"]:
        print(f"  {k:16s} jax {float(aux[k]):.6f}  torch {float(ref[k]):.6f}")

    if "grad_z" in ref:
        z_u, inv = np.asarray(dbg["z_u"]), np.asarray(dbg["inv"])
        gz = jax.grad(lambda z: model_losses(params, cfg, batch, max_unique, 0.01, 0.01, z_override=z)[0])(jnp.asarray(z_u))
        gz_torch = np.zeros_like(z_u)
        np.add.at(gz_torch, inv, ref["grad_z"])
        print(f"[grad_z] rel_err {rel_err(gz, gz_torch):.2e}  max|jax| {np.abs(np.asarray(gz)).max():.3e}  max|torch| {np.abs(gz_torch).max():.3e}")
        print(f"  torch rows with nonzero grad_z: {int((np.abs(ref['grad_z']).sum(1) > 0).sum())} of {len(inv)}; unique tasks {len(np.unique(inv))}")

    print("[grad]")
    errs = []
    n_degenerate = 0
    for k, g in grads.items():
        rk = "grad." + k
        if rk not in ref:
            continue
        if "layer_norms" in k and np.asarray(g).size == 1:
            # LayerNorm over a single element: normalized value is exactly 0,
            # so the true gradient of the affine weight is 0 and torch reports
            # rounding noise amplified by rstd
            n_degenerate += 1
            continue
        errs.append((rel_err(g, ref[rk]), k, float(np.abs(np.asarray(g)).max()), float(np.abs(ref[rk]).max())))
    errs.sort(reverse=True)
    for e, k, mj, mt in errs[:8]:
        print(f"  {e:.2e}  max|jax| {mj:.3e}  max|torch| {mt:.3e}  {k}")
    d_all = np.concatenate([np.ravel(np.asarray(grads[k], np.float64) - ref["grad." + k]) for _, k, _, _ in errs])
    t_all = np.concatenate([np.ravel(np.asarray(ref["grad." + k], np.float64)) for _, k, _, _ in errs])
    global_rel = float(np.linalg.norm(d_all) / np.linalg.norm(t_all))
    print(f"  global rel L2 error jax32 vs torch32: {global_rel:.2e}  (skipped {n_degenerate} degenerate size-1 LayerNorm weights)")
    worst_grad = errs[0][0]
    n_missing = sum(1 for k in grads if "grad." + k not in ref)
    print(f"  compared {len(errs)} tensors, {n_missing} without torch grad")
    if args.x64:
        p64 = {k: jnp.asarray(np.asarray(v), jnp.float64) for k, v in params.items()}
        b64 = [jnp.asarray(np.asarray(t), jnp.float64) for t in batch]

        def loss64(p):
            return model_losses(p, cfg, b64, max_unique, 0.01, 0.01)[0]

        g64 = jax.jit(jax.grad(loss64))(p64)
        print("[grad vs float64 reference]  columns: torch32/jax64, jax32/jax64")
        rows = []
        for k in grads:
            if "grad." + k not in ref or ("layer_norms" in k and np.asarray(grads[k]).size == 1):
                continue
            rows.append((rel_err(ref["grad." + k], g64[k]), rel_err(grads[k], g64[k]), k))
        rows.sort(reverse=True)
        for et, ej, k in rows[:8]:
            print(f"  {et:.2e}  {ej:.2e}  {k}")
        rows.sort(key=lambda r: r[1], reverse=True)
        print("  worst jax32/jax64:", f"{rows[0][1]:.2e}", rows[0][2])
        all_t = np.concatenate([np.ravel(np.asarray(ref["grad." + k], np.float64) - np.asarray(g64[k])) for _, _, k in rows])
        all_j = np.concatenate([np.ravel(np.asarray(grads[k], np.float64) - np.asarray(g64[k])) for _, _, k in rows])
        all_g = np.concatenate([np.ravel(np.asarray(g64[k])) for _, _, k in rows])
        print(f"  global rel L2 error: torch32 {np.linalg.norm(all_t) / np.linalg.norm(all_g):.2e}, jax32 {np.linalg.norm(all_j) / np.linalg.norm(all_g):.2e}")

    passed = worst_fwd < args.tol_fwd and global_rel < args.tol_grad
    print(f"RESULT forward worst {worst_fwd:.2e} grad global rel L2 {global_rel:.2e} (worst tensor {worst_grad:.2e}) -> {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
