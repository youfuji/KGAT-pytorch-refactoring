import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = REPO_ROOT / "scripts" / "profile_t_akdn_colab.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a profile matrix for T-AKDN on Colab."
    )
    parser.add_argument("--data_name", type=str, default="alibaba-fashion")
    parser.add_argument("--output_dir", type=str, default="profiling/matrix")
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--profile_steps", type=int, default=5)
    parser.add_argument("--att_chunk_size", type=int, default=131072)
    parser.add_argument("--cf_batch_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--relation_dim", type=int, default=64)
    parser.add_argument("--transr_dim", type=int, default=64)
    parser.add_argument("--edge_dropout_rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    return parser.parse_args()


def build_cases():
    return [
        {
            "name": "current",
            "flags": {
                "use_gru_lambda": 1,
                "use_dist_penalty": 1,
                "use_neighbor_zscore": 1,
                "use_concat_dist": 1,
                "use_lambda_annealing": 1,
            },
        },
        {
            "name": "no_dist",
            "flags": {
                "use_gru_lambda": 1,
                "use_dist_penalty": 0,
                "use_neighbor_zscore": 1,
                "use_concat_dist": 1,
                "use_lambda_annealing": 1,
            },
        },
        {
            "name": "no_gru_no_dist",
            "flags": {
                "use_gru_lambda": 0,
                "use_dist_penalty": 0,
                "use_neighbor_zscore": 1,
                "use_concat_dist": 1,
                "use_lambda_annealing": 1,
            },
        },
        {
            "name": "transr_dist",
            "flags": {
                "use_gru_lambda": 1,
                "use_dist_penalty": 1,
                "use_neighbor_zscore": 1,
                "use_concat_dist": 0,
                "use_lambda_annealing": 1,
            },
        },
    ]


def run_case(args, case, output_dir):
    output_path = output_dir / f"{args.data_name}_{case['name']}.json"
    cmd = [
        args.python_bin,
        str(PROFILE_SCRIPT),
        "--data_name", args.data_name,
        "--output_json", str(output_path),
        "--warmup_steps", str(args.warmup_steps),
        "--profile_steps", str(args.profile_steps),
        "--att_chunk_size", str(args.att_chunk_size),
        "--cf_batch_size", str(args.cf_batch_size),
        "--embed_dim", str(args.embed_dim),
        "--relation_dim", str(args.relation_dim),
        "--transr_dim", str(args.transr_dim),
        "--edge_dropout_rate", str(args.edge_dropout_rate),
        "--seed", str(args.seed),
    ]

    for key, value in case["flags"].items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n=== Running case: {case['name']} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return output_path


def load_json(path):
    return json.loads(Path(path).read_text())


def summarize_case(name, obj):
    phase = obj.get("phase_avg_ms", {})
    methods = obj.get("model_method_timing", {})
    summary = {
        "name": name,
        "iter_total_ms": phase.get("iter_total_ms"),
        "forward_ms": phase.get("forward_ms"),
        "backward_ms": phase.get("backward_ms"),
        "sample_ms": phase.get("sample_ms"),
        "calc_loss_ms": methods.get("calc_loss", {}).get("avg_ms"),
        "get_embeddings_ms": methods.get("get_embeddings", {}).get("avg_ms"),
        "_compute_kg_attention_ms": methods.get("_compute_kg_attention", {}).get("avg_ms"),
        "_compute_local_scores_ms": methods.get("_compute_local_scores", {}).get("avg_ms"),
        "_ig_aggregation_ms": methods.get("_ig_aggregation", {}).get("avg_ms"),
    }
    return summary


def print_summary_table(rows):
    headers = [
        "name",
        "iter_total_ms",
        "forward_ms",
        "backward_ms",
        "sample_ms",
        "calc_loss_ms",
        "get_embeddings_ms",
        "_compute_kg_attention_ms",
        "_compute_local_scores_ms",
        "_ig_aggregation_ms",
    ]

    col_widths = {}
    for h in headers:
        values = [h] + [str(row.get(h, "")) for row in rows]
        col_widths[h] = max(len(v) for v in values)

    print("\n=== Summary ===")
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers))


def main():
    args = parse_args()
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in build_cases():
        output_path = run_case(args, case, output_dir)
        obj = load_json(output_path)
        rows.append(summarize_case(case["name"], obj))

    print_summary_table(rows)

    summary_path = output_dir / f"{args.data_name}_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
