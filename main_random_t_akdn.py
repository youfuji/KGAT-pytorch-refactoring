import argparse
import logging
import random

import numpy as np
import torch
from tqdm import tqdm

from data_loader.loader_akdn import DataLoaderAKDN
from utils.metrics import calc_metrics_at_k


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a random-ranking baseline with the T-AKDN data pipeline."
    )
    parser.add_argument(
        "--data_name",
        nargs="?",
        default="alibaba-fashion",
        help="Dataset name. Default: alibaba-fashion",
    )
    parser.add_argument(
        "--data_dir",
        nargs="?",
        default="datasets/",
        help="Input data path.",
    )
    parser.add_argument(
        "--use_pretrain",
        type=int,
        default=0,
        help="Kept for DataLoader compatibility. Random baseline does not use pretraining.",
    )
    parser.add_argument(
        "--pretrain_embedding_dir",
        nargs="?",
        default="datasets/pretrain/",
        help="Kept for DataLoader compatibility.",
    )
    parser.add_argument(
        "--cf_batch_size",
        type=int,
        default=4096,
        help="Kept for DataLoader compatibility.",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=4096,
        help="Number of users to evaluate in each batch.",
    )
    parser.add_argument(
        "--Ks",
        nargs="?",
        default="[20]",
        help="List of K values, e.g. '[10, 20]'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2019,
        help="Base random seed.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="How many random trials to average over.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_random_baseline(dataloader, ks, seed):
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict
    test_batch_size = dataloader.test_batch_size

    user_ids = list(test_user_dict.keys())
    user_id_batches = [
        user_ids[i:i + test_batch_size]
        for i in range(0, len(user_ids), test_batch_size)
    ]
    item_ids = np.arange(dataloader.n_items, dtype=np.int64)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    metric_names = ["recall", "ndcg"]
    metrics_dict = {k: {m: [] for m in metric_names} for k in ks}

    with tqdm(total=len(user_id_batches), desc=f"Random Eval seed={seed}") as pbar:
        for batch_user_ids in user_id_batches:
            batch_scores = torch.rand(
                (len(batch_user_ids), dataloader.n_items),
                generator=rng,
                dtype=torch.float32,
            )
            batch_metrics = calc_metrics_at_k(
                batch_scores,
                train_user_dict,
                test_user_dict,
                np.asarray(batch_user_ids, dtype=np.int64),
                item_ids,
                ks,
            )

            for k in ks:
                for metric_name in metric_names:
                    metrics_dict[k][metric_name].append(batch_metrics[k][metric_name])
            pbar.update(1)

    for k in ks:
        for metric_name in metric_names:
            metrics_dict[k][metric_name] = np.concatenate(metrics_dict[k][metric_name]).mean()

    return metrics_dict


def main():
    args = parse_args()
    ks = sorted(eval(args.Ks))

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(args)

    set_seed(args.seed)
    data = DataLoaderAKDN(args, logging)

    all_run_metrics = {k: {"recall": [], "ndcg": []} for k in ks}

    for run_idx in range(args.n_runs):
        run_seed = args.seed + run_idx
        metrics = evaluate_random_baseline(data, ks, run_seed)
        summary = []
        for k in ks:
            recall = metrics[k]["recall"]
            ndcg = metrics[k]["ndcg"]
            all_run_metrics[k]["recall"].append(recall)
            all_run_metrics[k]["ndcg"].append(ndcg)
            summary.append(
                f"@{k}: Recall={recall:.4f}, NDCG={ndcg:.4f}"
            )
        logging.info("Run %02d (seed=%d) | %s", run_idx + 1, run_seed, " | ".join(summary))

    logging.info("Random baseline summary on %s", args.data_name)
    for k in ks:
        recall_values = np.asarray(all_run_metrics[k]["recall"], dtype=np.float32)
        ndcg_values = np.asarray(all_run_metrics[k]["ndcg"], dtype=np.float32)
        logging.info(
            "@%d | Recall mean=%.4f std=%.4f | NDCG mean=%.4f std=%.4f",
            k,
            recall_values.mean(),
            recall_values.std(),
            ndcg_values.mean(),
            ndcg_values.std(),
        )


if __name__ == "__main__":
    main()
