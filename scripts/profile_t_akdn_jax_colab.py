import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader.loader_akdn import DataLoaderAKDN
from main_t_akdn_jax import build_batch_dict, resolve_epoch_lambda
from model.t_akdn_jax import TAKDNJAX, build_jax_graph_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile JAX T-AKDN on Colab and export timing breakdowns."
    )
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--data_name", type=str, default="alibaba-fashion")
    parser.add_argument("--data_dir", type=str, default="datasets/")
    parser.add_argument("--use_pretrain", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--pretrain_embedding_dir", type=str, default="datasets/pretrain/")
    parser.add_argument("--pretrain_model_path", type=str, default="trained_model/model.npz")

    parser.add_argument("--cf_batch_size", type=int, default=4096)
    parser.add_argument("--test_batch_size", type=int, default=10000)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--relation_dim", type=int, default=64)
    parser.add_argument("--conv_dim_list", type=str, default="[64, 64, 64]")
    parser.add_argument("--mess_dropout", type=str, default="[0.1, 0.1, 0.1]")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.5)
    parser.add_argument("--cf_l2loss_lambda", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--transr_dim", type=int, default=64)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--use_gru_lambda", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_dist_penalty", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_neighbor_zscore", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_concat_dist", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_lambda_annealing", type=int, default=1, choices=[0, 1])
    parser.add_argument("--lambda_init", type=float, default=0.0)
    parser.add_argument("--lambda_final", type=float, default=0.5)
    parser.add_argument("--lambda_min", type=float, default=0.0)
    parser.add_argument("--lambda_max", type=float, default=1.0)
    parser.add_argument("--lambda_hidden_dim", type=int, default=64)
    parser.add_argument("--lambda_warmup_epochs", type=int, default=100)
    parser.add_argument("--lambda_anneal_epochs", type=int, default=400)
    parser.add_argument("--backend", type=str, default="jax")
    parser.add_argument("--jax_disable_jit", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eval_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float64"])

    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--profile_steps", type=int, default=5)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output_json", type=str, default="profiling/t_akdn_jax_profile.json")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "shape"):
        return np.asarray(obj).tolist()
    return obj


def main():
    args = parse_args()
    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)

    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = DataLoaderAKDN(args, logging)
    graph_data = build_jax_graph_data(data)

    if args.use_pretrain == 1:
        user_pre_embed = data.user_pre_embed
        item_pre_embed = data.item_pre_embed
    else:
        user_pre_embed, item_pre_embed = None, None

    model = TAKDNJAX(
        args,
        data.n_users,
        data.n_items,
        data.n_entities,
        data.n_relations,
        graph_data,
        user_pre_embed=user_pre_embed,
        item_pre_embed=item_pre_embed,
    )
    params = model.init_params(args.seed)
    opt_state = model.init_optimizer(params)
    rng = jax.random.PRNGKey(args.seed)
    lambda_value = resolve_epoch_lambda(args, 1)

    total_steps = args.max_batches if args.max_batches > 0 else args.warmup_steps + args.profile_steps
    profile_start_step = args.warmup_steps
    phase_totals = defaultdict(float)
    iter_summaries = []
    profiled_steps = 0

    for step in range(total_steps):
        summary = {"step": step}

        t0 = time.perf_counter()
        cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(
            data.train_user_dict, data.cf_batch_size
        )
        summary["sample_ms"] = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        batch = build_batch_dict(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
        batch = jax.tree_util.tree_map(jax.device_put, batch)
        summary["h2d_ms"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        rng, step_rng = jax.random.split(rng)
        params, opt_state, aux = model.train_step(params, opt_state, batch, step_rng, lambda_value)
        aux = jax.tree_util.tree_map(jax.device_get, aux)
        summary["train_step_ms"] = (time.perf_counter() - t2) * 1000.0
        summary["loss"] = float(aux["loss"])
        summary["lambda_records"] = np.asarray(aux["lambda_records"]).tolist()
        summary["iter_total_ms"] = summary["sample_ms"] + summary["h2d_ms"] + summary["train_step_ms"]

        iter_summaries.append(summary)
        if step >= profile_start_step:
            profiled_steps += 1
            for name in ["sample_ms", "h2d_ms", "train_step_ms", "iter_total_ms"]:
                phase_totals[name] += summary[name]

        logging.info(
            "Step %d | loss=%.4f | sample=%.2fms | h2d=%.2fms | train=%.2fms | total=%.2fms",
            step,
            summary["loss"],
            summary["sample_ms"],
            summary["h2d_ms"],
            summary["train_step_ms"],
            summary["iter_total_ms"],
        )

    phase_avg_ms = {
        name: round(total / max(profiled_steps, 1), 3)
        for name, total in phase_totals.items()
    }
    result = {
        "devices": [str(device) for device in jax.devices()],
        "profiled_steps": profiled_steps,
        "phase_avg_ms": phase_avg_ms,
        "iterations": iter_summaries,
    }

    output_path = REPO_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_jsonable(result), indent=2), encoding="utf-8")
    logging.info("Saved profiling report to %s", output_path)


if __name__ == "__main__":
    main()
