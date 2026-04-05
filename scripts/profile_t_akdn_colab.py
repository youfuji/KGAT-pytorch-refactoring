import argparse
import json
import logging
import random
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader.loader_akdn import DataLoaderAKDN
from model.T_AKDN import T_AKDN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile T-AKDN on Colab and export timing breakdowns."
    )

    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--data_name", type=str, default="alibaba-fashion")
    parser.add_argument("--data_dir", type=str, default="datasets/")
    parser.add_argument("--use_pretrain", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--pretrain_embedding_dir", type=str, default="datasets/pretrain/")
    parser.add_argument("--pretrain_model_path", type=str, default="trained_model/model.pth")

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
    parser.add_argument("--use_transr_attention", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_tau_softmax", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_dist_penalty", type=int, default=1, choices=[0, 1])
    parser.add_argument("--lambda_mode", type=str, default="glu",
                        choices=["anneal", "glu", "fixed"])
    parser.add_argument("--score_norm_mode", type=str, default="neighbor_zscore",
                        choices=["neighbor_zscore", "global_zscore", "global_minmax"])
    parser.add_argument("--use_concat_dist", type=int, default=1, choices=[0, 1])
    parser.add_argument("--att_chunk_size", type=int, default=0)
    parser.add_argument("--lambda_init", type=float, default=0.0)
    parser.add_argument("--lambda_final", type=float, default=1.0)
    parser.add_argument("--lambda_warmup_epochs", type=int, default=100)
    parser.add_argument("--lambda_anneal_epochs", type=int, default=400)
    parser.add_argument("--lambda_min", type=float, default=0.0)
    parser.add_argument("--lambda_max", type=float, default=1.0)
    parser.add_argument("--lambda_glu_hidden_dim", type=int, default=64)

    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--profile_steps", type=int, default=5)
    parser.add_argument("--max_batches", type=int, default=0,
                        help="Hard cap on training iterations. 0 means warmup_steps + profile_steps.")
    parser.add_argument("--output_json", type=str, default="profiling/t_akdn_profile.json")
    parser.add_argument("--log_every", type=int, default=1)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


class TimerStore:
    def __init__(self):
        self.stats = defaultdict(lambda: {"count": 0, "total_ms": 0.0})

    def add(self, name, elapsed_ms):
        stat = self.stats[name]
        stat["count"] += 1
        stat["total_ms"] += elapsed_ms

    def summary(self):
        result = {}
        for name, stat in sorted(self.stats.items()):
            count = stat["count"]
            total_ms = stat["total_ms"]
            result[name] = {
                "count": count,
                "total_ms": round(total_ms, 3),
                "avg_ms": round(total_ms / count, 3) if count else 0.0,
            }
        return result


def wrap_timed_method(model, timer_store, device, method_name):
    if not hasattr(model, method_name):
        return

    original = getattr(model, method_name)
    if not callable(original):
        return

    def wrapped(self, *args, **kwargs):
        sync_if_needed(device)
        start = time.perf_counter()
        out = original(*args, **kwargs)
        sync_if_needed(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timer_store.add(method_name, elapsed_ms)
        return out

    setattr(model, method_name, types.MethodType(wrapped, model))


def attach_model_profiling(model, timer_store, device):
    method_names = [
        "_compute_edge_lambda",
        "_compute_local_scores_transr",
        "_compute_local_scores_akdn",
        "_compute_local_scores",
        "_compute_kg_attention_full",
        "_compute_kg_attention_chunked",
        "_compute_kg_attention",
        "_compute_concat_dist",
        "_compute_transr_dist",
        "_fuse_attention_sem_only",
        "_fuse_attention_with_dist",
        "_edge_softmax",
        "_neighbor_zscore",
        "_global_zscore",
        "_global_minmax",
        "_kg_aggregation_full",
        "_kg_aggregation_chunked",
        "_ig_aggregation",
        "fusion_gate",
        "get_embeddings",
        "calc_loss",
    ]
    for method_name in method_names:
        wrap_timed_method(model, timer_store, device, method_name)


def build_model(args, data, device):
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    model = T_AKDN(
        args,
        data.n_users,
        data.n_items,
        data.n_entities,
        data.n_relations,
        A_in=data.norm_adj_mat,
        user_pre_embed=user_pre_embed,
        item_pre_embed=item_pre_embed,
        edge_dropout_rate=args.edge_dropout_rate,
    )
    model.to(device)

    relations = list(data.train_relation_dict.keys())
    model.set_kg_structure(
        data.h_list.to(device),
        data.t_list.to(device),
        data.r_list.to(device),
        relations,
    )
    return model


def get_scheduled_lambda(args, epoch):
    if args.lambda_mode == "glu":
        return None
    if args.lambda_mode == "fixed":
        return args.lambda_final
    if epoch <= args.lambda_warmup_epochs:
        return args.lambda_init

    anneal_progress = epoch - args.lambda_warmup_epochs
    if anneal_progress >= args.lambda_anneal_epochs:
        return args.lambda_final

    ratio = anneal_progress / max(args.lambda_anneal_epochs, 1)
    return args.lambda_init + ratio * (args.lambda_final - args.lambda_init)


def profile_train_loop(args):
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DataLoaderAKDN(args, logging)
    model = build_model(args, data, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model_timer = TimerStore()
    attach_model_profiling(model, model_timer, device)

    total_steps = args.max_batches if args.max_batches > 0 else args.warmup_steps + args.profile_steps
    profile_start_step = args.warmup_steps

    iter_summaries = []
    phase_totals = defaultdict(float)
    profiled_steps = 0

    model.train()

    for step in range(total_steps):
        summary = {"step": step}
        epoch = step + 1
        if args.use_dist_penalty and args.lambda_mode != "glu":
            model.set_lambda(get_scheduled_lambda(args, epoch))

        t0 = time.perf_counter()
        cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(
            data.train_user_dict, data.cf_batch_size
        )
        summary["sample_ms"] = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        cf_batch_user = cf_batch_user.to(device)
        cf_batch_pos_item = cf_batch_pos_item.to(device)
        cf_batch_neg_item = cf_batch_neg_item.to(device)
        sync_if_needed(device)
        summary["h2d_ms"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        batch_loss = model("calc_loss", cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
        sync_if_needed(device)
        summary["forward_ms"] = (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        batch_loss.backward()
        sync_if_needed(device)
        summary["backward_ms"] = (time.perf_counter() - t3) * 1000.0

        t4 = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync_if_needed(device)
        summary["optim_ms"] = (time.perf_counter() - t4) * 1000.0

        summary["loss"] = float(batch_loss.detach().cpu())
        summary["iter_total_ms"] = (
            summary["sample_ms"]
            + summary["h2d_ms"]
            + summary["forward_ms"]
            + summary["backward_ms"]
            + summary["optim_ms"]
        )

        if device.type == "cuda":
            summary["cuda_max_allocated_mb"] = round(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2
            )
            summary["cuda_max_reserved_mb"] = round(
                torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2
            )

        iter_summaries.append(summary)

        if step >= profile_start_step:
            profiled_steps += 1
            for key in ["sample_ms", "h2d_ms", "forward_ms", "backward_ms", "optim_ms", "iter_total_ms"]:
                phase_totals[key] += summary[key]

        if args.log_every > 0 and ((step + 1) % args.log_every == 0):
            logging.info(
                "step=%d loss=%.4f total=%.1fms sample=%.1f h2d=%.1f fwd=%.1f bwd=%.1f opt=%.1f",
                step,
                summary["loss"],
                summary["iter_total_ms"],
                summary["sample_ms"],
                summary["h2d_ms"],
                summary["forward_ms"],
                summary["backward_ms"],
                summary["optim_ms"],
            )

    avg_phase = {
        key: round(value / profiled_steps, 3) if profiled_steps else 0.0
        for key, value in sorted(phase_totals.items())
    }

    result = {
        "environment": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "data_name": args.data_name,
        },
        "config": {
            "cf_batch_size": args.cf_batch_size,
            "embed_dim": args.embed_dim,
            "relation_dim": args.relation_dim,
            "transr_dim": args.transr_dim,
            "att_chunk_size": args.att_chunk_size,
            "use_transr_attention": args.use_transr_attention,
            "use_tau_softmax": args.use_tau_softmax,
            "use_dist_penalty": args.use_dist_penalty,
            "lambda_mode": args.lambda_mode,
            "score_norm_mode": args.score_norm_mode,
            "use_concat_dist": args.use_concat_dist,
            "lambda_init": args.lambda_init,
            "lambda_final": args.lambda_final,
            "lambda_warmup_epochs": args.lambda_warmup_epochs,
            "lambda_anneal_epochs": args.lambda_anneal_epochs,
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
            "lambda_glu_hidden_dim": args.lambda_glu_hidden_dim,
            "edge_dropout_rate": args.edge_dropout_rate,
        },
        "dataset": {
            "n_users": data.n_users,
            "n_items": data.n_items,
            "n_entities": data.n_entities,
            "n_relations": data.n_relations,
            "n_cf_train": data.n_cf_train,
            "n_kg_train": data.n_kg_train,
        },
        "run": {
            "warmup_steps": args.warmup_steps,
            "profile_steps": args.profile_steps,
            "total_steps": total_steps,
            "lambda_records": model.lambda_records,
        },
        "phase_avg_ms": avg_phase,
        "model_method_timing": model_timer.summary(),
        "iterations": iter_summaries,
    }
    result = to_jsonable(result)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    logging.info("saved profile to %s", output_path)
    logging.info("phase averages (ms): %s", json.dumps(avg_phase, indent=2))


if __name__ == "__main__":
    profile_train_loop(parse_args())
