"""
HyPoGen in JAX as pure functions over a flat parameter dict.

Parameter keys and tensor layouts mirror the torch state_dict of
models.rl_regressor.HyperRLSolution (hyper_type=hypogen), so a torch
checkpoint can be loaded without any renaming. Linear weights are
stored as (out, in) and applied as x @ W.T + b, as in torch.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np


LN_EPS = 1e-5
ROOT = "hyper_rl_net"


@dataclass
class HyPoGenConfig:
    input_dim: int
    state_dim: int
    action_dim: int
    embed_dim: int = 256
    hidden_dim: int = 256
    # widths of the policy head's hidden layers; empty means [hidden_dim],
    # i.e. the upstream single-hidden-layer net
    hidden_dims: tuple = ()
    # same for the critic head; empty falls back to hidden_dims, so one knob
    # covers both heads unless they are deliberately split
    critic_hidden_dims: tuple = ()
    weight_dim: int = 128
    enc_dec_dim: int = 256
    num_enc_dec_layer: int = 2
    opt_block_dim: int = 128
    num_opt_mlp_layer: int = 2
    num_layers: int = 8
    pseudo_dim: int = 64
    dl_din_way: str = "slice"
    dl_dw_way: str = "direct"
    # hidden nonlinearity of the policy head; the critic head stays relu
    policy_act: str = "relu"
    init_lr: float = -1e-2
    per_task_apply: bool = False

    def head_layers(self, head: int) -> List[tuple]:
        # (name, in_dim, out_dim) of the target net layers of a head
        in_dim = self.state_dim if head == 1 else self.state_dim + self.action_dim
        out_dim = self.action_dim if head == 1 else 1
        dims = self.hidden_dims if head == 1 else (self.critic_hidden_dims
                                                   or self.hidden_dims)
        widths = [int(w) for w in dims] or [self.hidden_dim]
        dims = [in_dim] + widths
        return ([(f"fc{i}", dims[i], dims[i + 1]) for i in range(len(widths))]
                + [(f"fc{len(widths)}", widths[-1], out_dim)])

    def pseudo_dims(self, head: int):
        layers = self.head_layers(head)
        in_dims = [layers[0][1]] + [self.pseudo_dim] * (len(layers) - 1)
        out_dims = [self.pseudo_dim] * (len(layers) - 1) + [layers[-1][2]]
        return in_dims, out_dims


def _linear(p, prefix, x):
    return x @ p[prefix + ".weight"].T + p[prefix + ".bias"]


def _mlp(p, prefix, x, n_layers):
    for i in range(n_layers):
        x = _linear(p, f"{prefix}.fcs.{i}", x)
        if i < n_layers - 1:
            x = jax.nn.relu(x)
    return x


def _resblock(p, prefix, x):
    h = _linear(p, prefix + ".fc.1", jax.nn.relu(x))
    h = _linear(p, prefix + ".fc.3", jax.nn.relu(h))
    return x + h


def task_embedding(p, meta_v):
    pre = f"{ROOT}.hyper.hyper"
    x = _linear(p, f"{pre}.0", meta_v)
    x = _resblock(p, f"{pre}.1", x)
    x = _resblock(p, f"{pre}.2", x)
    x = _linear(p, f"{pre}.3", x)
    x = _resblock(p, f"{pre}.4", x)
    x = _resblock(p, f"{pre}.5", x)
    x = _linear(p, f"{pre}.6", x)
    x = _resblock(p, f"{pre}.7", x)
    x = _resblock(p, f"{pre}.8", x)
    return x


def _layer_shapes(cfg: HyPoGenConfig, head: int):
    # ordered (weight, bias) shapes per target layer, matching named_parameters
    return [
        {"weight": (out_dim, in_dim), "bias": (out_dim,)}
        for _, in_dim, out_dim in cfg.head_layers(head)
    ]


def _encode(p, prefix, wd, n_layers):
    vec = jnp.concatenate(
        [wd["weight"].reshape(-1, wd["weight"].shape[-1] * wd["weight"].shape[-2]),
         wd["bias"].reshape(-1, wd["bias"].shape[-1])],
        axis=-1,
    )
    return _mlp(p, prefix, vec, n_layers)


def _decode(p, prefix, vec, shapes, n_layers):
    out = _mlp(p, prefix, vec, n_layers)
    n_w = int(np.prod(shapes["weight"]))
    w, b = out[:, :n_w], out[:, n_w:]
    return {"weight": w.reshape(-1, *shapes["weight"]), "bias": b.reshape(-1, *shapes["bias"])}


def _param_ln(p, prefix, wd):
    out = {}
    for j, (k, v) in enumerate(wd.items()):
        axes = tuple(range(1, v.ndim))
        mean = v.mean(axis=axes, keepdims=True)
        var = ((v - mean) ** 2).mean(axis=axes, keepdims=True)
        y = (v - mean) / jnp.sqrt(var + LN_EPS)
        out[k] = y * p[f"{prefix}.ln.{j}.weight"] + p[f"{prefix}.ln.{j}.bias"]
    return out


def _opt_block(p, cfg: HyPoGenConfig, head: int, n: int, ftask, w_dicts):
    hp = f"{ROOT}.hyper_mlp{head}"
    bp = f"{hp}.opt_blocks.{n}"
    in_dims, out_dims = cfg.pseudo_dims(head)
    shapes = _layer_shapes(cfg, head)
    n_u = ftask.shape[0]
    nl = cfg.num_opt_mlp_layer

    z_ins = [_mlp(p, f"{bp}.forward_in", ftask, nl)]
    dl_douts = [_mlp(p, f"{bp}.dloss_dout", ftask, nl)]

    embs = []
    for l, wd in enumerate(w_dicts):
        e = _encode(p, f"{hp}.encoders.{l}.encoder", wd, cfg.num_enc_dec_layer)
        embs.append(jnp.broadcast_to(e, (n_u, e.shape[-1])))

    for l, e in enumerate(embs):
        sp = f"{bp}.opt_sub_blocks.{l}"
        z_ins.append(_mlp(p, f"{sp}.forward_net", jnp.concatenate([z_ins[-1], e], -1), nl))

    dw_dicts = [None] * len(embs)
    for l in reversed(range(len(embs))):
        sp = f"{bp}.opt_sub_blocks.{l}"
        x = jnp.concatenate([z_ins[l], embs[l]], -1)
        dl_dout = dl_douts[-1]
        in_l, out_l = in_dims[l], out_dims[l]
        dout_din = _mlp(p, f"{sp}.dout_din", x, nl).reshape(n_u, in_l, out_l)
        dout_dw = _mlp(p, f"{sp}.dout_dw", x, nl).reshape(n_u, cfg.weight_dim, out_l)

        if cfg.dl_din_way == "direct":
            dl_din = jnp.einsum("no,nio->ni", dl_dout, dout_din)
        elif cfg.dl_din_way == "slice":
            rep = jnp.broadcast_to(dl_dout[:, None, :], (n_u, in_l, out_l))
            dl_din = _mlp(p, f"{sp}.mm_mlp_in", jnp.concatenate([rep, dout_din], -1), nl)[..., 0]
        else:
            raise NotImplementedError(cfg.dl_din_way)

        if cfg.dl_dw_way == "direct":
            dl_dw = jnp.einsum("no,nio->ni", dl_dout, dout_dw)
        elif cfg.dl_dw_way == "slice":
            rep = jnp.broadcast_to(dl_dout[:, None, :], (n_u, cfg.weight_dim, out_l))
            dl_dw = _mlp(p, f"{sp}.mm_mlp_w", jnp.concatenate([rep, dout_dw], -1), nl)[..., 0]
        else:
            raise NotImplementedError(cfg.dl_dw_way)

        dl_douts.append(dl_din)
        dw_dicts[l] = _decode(p, f"{hp}.decoders.{l}.decoder", dl_dw, shapes[l], cfg.num_enc_dec_layer)
    return dw_dicts


def forward_weights(p, cfg: HyPoGenConfig, head: int, ftask) -> List[Dict[str, jnp.ndarray]]:
    """Run the K optimization blocks; returns one merged weight dict per block."""
    hp = f"{ROOT}.hyper_mlp{head}"
    names = [name for name, _, _ in cfg.head_layers(head)]
    w_dicts = [
        {"weight": p[f"{hp}.target_net.{name}.weight"], "bias": p[f"{hp}.target_net.{name}.bias"]}
        for name in names
    ]
    lrs = p[f"{hp}.dynamic_lrs"]
    finals = []
    for n in range(cfg.num_layers):
        upd = _opt_block(p, cfg, head, n, ftask, w_dicts)
        upd = [_param_ln(p, f"{hp}.layer_norms.{l}", d) for l, d in enumerate(upd)]
        w_dicts = [
            {k: wd[k] + lrs[n] * ud[k] for k in wd} for wd, ud in zip(w_dicts, upd)
        ]
        finals.append({f"{name}.{k}": v for name, wd in zip(names, w_dicts) for k, v in wd.items()})
    return finals


_ACTS = {"relu": jax.nn.relu, "tanh": jnp.tanh}


def head_act(cfg: HyPoGenConfig, head: int):
    """Hidden nonlinearity of a head's target net."""
    if head != 1:
        return jax.nn.relu
    try:
        return _ACTS[cfg.policy_act]
    except KeyError:
        raise ValueError(f"policy_act={cfg.policy_act!r} is not "
                         f"{sorted(_ACTS)}") from None


def apply_target(cfg: HyPoGenConfig, head: int, w, x):
    """Per-sample target MLP: w entries are (B, ...) and x is (B, in)."""
    names = [name for name, _, _ in cfg.head_layers(head)]
    act = head_act(cfg, head)
    h = x
    for i, name in enumerate(names):
        h = jnp.einsum("bi,boi->bo", h, w[f"{name}.weight"]) + w[f"{name}.bias"]
        if i < len(names) - 1:
            h = act(h)
    return jnp.tanh(h) if head == 1 else h


def apply_target_per_task(cfg: HyPoGenConfig, head: int, w, x, inv):
    """Task-major target MLP: w entries are (U, ...) and x is (B, in).

    Same maths as gathering the weights per sample, but matmuls instead of
    batched matrix-vector products, at a factor U more FLOPs.
    """
    names = [name for name, _, _ in cfg.head_layers(head)]
    act = head_act(cfg, head)
    h = None
    for i, name in enumerate(names):
        weight, bias = w[f"{name}.weight"], w[f"{name}.bias"]
        h = (
            jnp.einsum("bi,uoi->ubo", x, weight)
            if h is None
            else jnp.einsum("ubi,uoi->ubo", h, weight)
        ) + bias[:, None, :]
        if i < len(names) - 1:
            h = act(h)
    h = jnp.tanh(h) if head == 1 else h
    return h[inv, jnp.arange(h.shape[1])]


def _apply(cfg: HyPoGenConfig, head: int, w, x, inv):
    if cfg.per_task_apply:
        return apply_target_per_task(cfg, head, w, x, inv)
    return apply_target(cfg, head, gather(w, inv), x)


def unique_tasks(input_param, max_unique):
    uniq, inv = jnp.unique(
        input_param, axis=0, size=max_unique, fill_value=0.0, return_inverse=True
    )
    return uniq, inv.reshape(-1)


def gather(w, inv):
    return {k: v[inv] for k, v in w.items()}


def model_losses(
    p, cfg: HyPoGenConfig, batch, max_unique, value_weight, td_weight,
    z_override=None, task=None,
):
    """Training loss of approximators.rl_solution.RLApproximator.update.

    task is an optional precomputed (uniq, inv), avoiding a per-step sort.
    """
    input_param, state, action, next_state, reward, discount, value = batch
    uniq, inv = unique_tasks(input_param, max_unique) if task is None else task
    z_u = task_embedding(p, uniq) if z_override is None else z_override
    pol_w = forward_weights(p, cfg, 1, z_u)
    q_w = forward_weights(p, cfg, 2, z_u)

    sa = jnp.concatenate([state, action], -1)
    pred_action = jnp.stack([_apply(cfg, 1, w, state, inv) for w in pol_w])
    pred_q = jnp.stack([_apply(cfg, 2, w, sa, inv) for w in q_w])

    loss_action = jnp.mean((pred_action - action[None]) ** 2)
    loss_value = jnp.mean((pred_q - value[None]) ** 2)

    next_action = jax.lax.stop_gradient(_apply(cfg, 1, pol_w[-1], next_state, inv))
    nsa = jnp.concatenate([next_state, next_action], -1)
    target_q = reward + discount * _apply(cfg, 2, q_w[-1], nsa, inv)
    loss_td = jnp.mean((value - target_q) ** 2)

    loss = loss_action + value_weight * loss_value + td_weight * loss_td
    aux = {
        "loss_total": loss,
        "loss_action_pred": loss_action,
        "loss_value_pred": value_weight * loss_value,
        "loss_td": td_weight * loss_td,
    }
    debug = {
        "z": z_u[inv],
        "z_u": z_u,
        "inv": inv,
        "pred_action": pred_action,
        "pred_q": pred_q,
        "next_action": next_action,
        "target_q": target_q,
    }
    return loss, (aux, debug)


def predict_action(p, cfg: HyPoGenConfig, input_param, state, max_unique):
    """Zero-shot policy output using the last optimization step."""
    uniq, inv = unique_tasks(input_param, max_unique)
    z_u = task_embedding(p, uniq)
    w = forward_weights(p, cfg, 1, z_u)[-1]
    return _apply(cfg, 1, w, state, inv)


def _uniform(key, shape, bound):
    return jax.random.uniform(key, shape, jnp.float32, -bound, bound)


def init_params(key, cfg: HyPoGenConfig) -> Dict[str, jnp.ndarray]:
    """Torch-matching initialization of every tensor in the flat param dict."""
    params = {}
    keys = iter(jax.random.split(key, 4096))

    def linear(prefix, in_dim, out_dim, w_bound=None):
        bound = 1.0 / math.sqrt(in_dim)
        params[prefix + ".weight"] = _uniform(next(keys), (out_dim, in_dim), w_bound or bound)
        params[prefix + ".bias"] = _uniform(next(keys), (out_dim,), bound)

    def mlp(prefix, in_dim, out_dim, hidden, n_layers):
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        for i in range(n_layers):
            linear(f"{prefix}.fcs.{i}", dims[i], dims[i + 1])

    # task embedding: torch Meta_Embedding.init_layers halves the weight bound
    pre = f"{ROOT}.hyper.hyper"
    d = cfg.embed_dim
    plan = [(0, cfg.input_dim, d // 4), (3, d // 4, d // 2), (6, d // 2, d)]
    for idx, i_dim, o_dim in plan:
        linear(f"{pre}.{idx}", i_dim, o_dim, 1.0 / (2.0 * math.sqrt(i_dim)))
        for r in (idx + 1, idx + 2):
            linear(f"{pre}.{r}.fc.1", o_dim, o_dim, 1.0 / (2.0 * math.sqrt(o_dim)))
            linear(f"{pre}.{r}.fc.3", o_dim, o_dim, 1.0 / (2.0 * math.sqrt(o_dim)))

    for head in (1, 2):
        hp = f"{ROOT}.hyper_mlp{head}"
        params[f"{hp}.dynamic_lrs"] = jnp.full((cfg.num_layers,), cfg.init_lr, jnp.float32)
        layers = cfg.head_layers(head)
        shapes = _layer_shapes(cfg, head)
        in_dims, out_dims = cfg.pseudo_dims(head)
        for name, i_dim, o_dim in layers:
            linear(f"{hp}.target_net.{name}", i_dim, o_dim)
        for l, sh in enumerate(shapes):
            cnt = int(np.prod(sh["weight"]) + np.prod(sh["bias"]))
            mlp(f"{hp}.encoders.{l}.encoder", cnt, cfg.weight_dim, cfg.enc_dec_dim, cfg.num_enc_dec_layer)
            mlp(f"{hp}.decoders.{l}.decoder", cfg.weight_dim, cnt, cfg.enc_dec_dim, cfg.num_enc_dec_layer)
            for j, k in enumerate(("weight", "bias")):
                params[f"{hp}.layer_norms.{l}.ln.{j}.weight"] = jnp.ones(sh[k], jnp.float32)
                params[f"{hp}.layer_norms.{l}.ln.{j}.bias"] = jnp.zeros(sh[k], jnp.float32)
        for n in range(cfg.num_layers):
            bp = f"{hp}.opt_blocks.{n}"
            nl, hd, wd = cfg.num_opt_mlp_layer, cfg.opt_block_dim, cfg.weight_dim
            for l, (in_l, out_l) in enumerate(zip(in_dims, out_dims)):
                sp = f"{bp}.opt_sub_blocks.{l}"
                mlp(f"{sp}.forward_net", in_l + wd, out_l, hd, nl)
                mlp(f"{sp}.dout_din", in_l + wd, out_l * in_l, hd, nl)
                mlp(f"{sp}.dout_dw", in_l + wd, out_l * wd, hd, nl)
                if cfg.dl_din_way == "slice":
                    mlp(f"{sp}.mm_mlp_in", 2 * out_l, 1, hd, nl)
                if cfg.dl_dw_way == "slice":
                    mlp(f"{sp}.mm_mlp_w", 2 * out_l, 1, hd, nl)
            mlp(f"{bp}.forward_in", cfg.embed_dim, in_dims[0], hd, nl)
            mlp(f"{bp}.dloss_dout", cfg.embed_dim, out_dims[-1], hd, nl)
    return params


def count_params(params):
    return int(sum(np.prod(v.shape) for v in params.values()))


def load_npz(path):
    with np.load(path) as f:
        return {k: jnp.asarray(f[k]) for k in f.files}


def save_npz(path, params):
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})
