import json
import logging
import os
import random
from time import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_loader.loader_akdn import DataLoaderAKDN
from model.t_akdn_jax import TAKDNJAX, build_jax_graph_data
from parser.parser_t_akdn import parse_t_akdn_args, resolve_t_akdn_save_dir
from utils.log_helper import create_log_id, logging_config
from utils.metrics import calc_metrics_at_k
from utils.model_helper import early_stopping


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_lambda_records(lambda_records):
    if len(lambda_records) == 0:
        return "[]"
    return "[" + ", ".join("{:.4f}".format(float(v)) for v in lambda_records) + "]"


def flatten_tree(tree, prefix=""):
    flat = {}
    for key, value in tree.items():
        current = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_tree(value, current))
        else:
            flat[current] = np.asarray(value)
    return flat


def unflatten_tree(flat):
    root = {}
    for path, value in flat.items():
        parts = path.split("/")
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = jnp.asarray(value)
    return root


def save_jax_checkpoint(params, model_dir, current_epoch, last_best_epoch=None):
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "model_epoch{}.npz".format(current_epoch))
    payload = flatten_tree(params)
    payload["epoch"] = np.asarray(current_epoch, dtype=np.int32)
    np.savez_compressed(ckpt_path, **payload)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_ckpt = os.path.join(model_dir, "model_epoch{}.npz".format(last_best_epoch))
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def load_jax_checkpoint(model_path):
    with np.load(model_path, allow_pickle=False) as data:
        flat = {k: data[k] for k in data.files if k != "epoch"}
        epoch = int(data["epoch"]) if "epoch" in data.files else -1
    return unflatten_tree(flat), epoch


def resolve_epoch_lambda(args, epoch):
    if not args.use_dist_penalty:
        return 0.0
    if args.use_gru_lambda:
        return args.lambda_init
    if args.use_lambda_annealing:
        if epoch <= args.lambda_warmup_epochs:
            return args.lambda_init
        if epoch <= args.lambda_warmup_epochs + args.lambda_anneal_epochs:
            progress = (epoch - args.lambda_warmup_epochs) / args.lambda_anneal_epochs
            return args.lambda_init + (args.lambda_final - args.lambda_init) * progress
    return args.lambda_final


def build_batch_dict(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item):
    return {
        "user_ids": jnp.asarray(cf_batch_user.cpu().numpy(), dtype=jnp.int32),
        "item_pos_ids": jnp.asarray(cf_batch_pos_item.cpu().numpy(), dtype=jnp.int32),
        "item_neg_ids": jnp.asarray(cf_batch_neg_item.cpu().numpy(), dtype=jnp.int32),
    }


def evaluate(model, params, dataloader, Ks, lambda_value):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i : i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    item_ids = jnp.arange(dataloader.n_items, dtype=jnp.int32)
    all_embed = model.compute_embeddings(params, jax.random.PRNGKey(0), lambda_value, training=False, return_aux=False)

    cf_scores = []
    metric_names = ["recall", "ndcg"]
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc="Evaluating Iteration") as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids_np = np.asarray(batch_user_ids, dtype=np.int32)
            batch_scores = model.score_from_embeddings(all_embed, jnp.asarray(batch_user_ids_np), item_ids)
            batch_scores = np.asarray(jax.device_get(batch_scores))
            batch_metrics = calc_metrics_at_k(
                batch_scores,
                train_user_dict,
                test_user_dict,
                batch_user_ids_np,
                np.arange(dataloader.n_items),
                Ks,
            )

            cf_scores.append(batch_scores)
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict


def train(args):
    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)

    if args.use_pretrain == 2:
        raise ValueError("JAX backend does not load PyTorch .pth checkpoints. Use --eval_only with a JAX .npz checkpoint.")

    set_seed(args.seed)
    args.backend = "jax"
    args.save_dir = resolve_t_akdn_save_dir(args)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name="log{:d}".format(log_save_id), no_console=False)
    logging.info(args)
    logging.info("JAX devices: %s", jax.devices())

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

    if args.eval_only:
        params, _ = load_jax_checkpoint(args.pretrain_model_path)
        Ks = eval(args.Ks)
        lambda_value = resolve_epoch_lambda(args, args.n_epoch)
        _, metrics_dict = evaluate(model, params, data, Ks, lambda_value)
        k_min = min(Ks)
        k_max = max(Ks)
        logging.info(
            "CF Evaluation: Recall [%.4f, %.4f], NDCG [%.4f, %.4f]",
            metrics_dict[k_min]["recall"],
            metrics_dict[k_max]["recall"],
            metrics_dict[k_min]["ndcg"],
            metrics_dict[k_max]["ndcg"],
        )
        return

    params = model.init_params(args.seed)
    opt_state = model.init_optimizer(params)

    best_epoch = -1
    best_recall = 0
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)
    epoch_list = []
    metrics_list = {k: {"recall": [], "ndcg": []} for k in Ks}

    base_rng = jax.random.PRNGKey(args.seed)

    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        total_loss = 0.0
        n_batch = data.n_cf_train // data.cf_batch_size + 1
        lambda_value = resolve_epoch_lambda(args, epoch)

        time_cf = time()
        for step in range(1, n_batch + 1):
            time_iter = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(
                data.train_user_dict, data.cf_batch_size
            )
            batch = build_batch_dict(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
            base_rng, step_rng = jax.random.split(base_rng)
            params, opt_state, aux = model.train_step(params, opt_state, batch, step_rng, lambda_value)

            batch_loss = float(aux["loss"])
            if np.isnan(batch_loss):
                raise FloatingPointError(
                    "ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.".format(epoch, step, n_batch)
                )

            total_loss += batch_loss
            lambda_log = format_lambda_records(np.asarray(aux["lambda_records"]))
            if (step % args.cf_print_every) == 0:
                logging.info(
                    "CF Training: Epoch %04d Iter %04d / %04d | Time %.1fs | Iter Loss %.4f | Iter Mean Loss %.4f | Tau %.4f | Lambda %s",
                    epoch,
                    step,
                    n_batch,
                    time() - time_iter,
                    batch_loss,
                    total_loss / step,
                    model.tau,
                    lambda_log,
                )

        logging.info(
            "CF Training: Epoch %04d Total Iter %04d | Total Time %.1fs | Iter Mean Loss %.4f | Tau %.4f | Lambda %s",
            epoch,
            n_batch,
            time() - time_cf,
            total_loss / n_batch,
            model.tau,
            format_lambda_records(np.asarray(aux["lambda_records"])),
        )
        logging.info("Epoch %04d finished | Total Time %.1fs", epoch, time() - time0)

        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time_eval = time()
            _, metrics_dict = evaluate(model, params, data, Ks, lambda_value)
            logging.info(
                "CF Evaluation: Epoch %04d | Total Time %.1fs | Recall [%.4f, %.4f], NDCG [%.4f, %.4f]",
                epoch,
                time() - time_eval,
                metrics_dict[k_min]["recall"],
                metrics_dict[k_max]["recall"],
                metrics_dict[k_min]["ndcg"],
                metrics_dict[k_max]["ndcg"],
            )

            epoch_list.append(epoch)
            for k in Ks:
                for metric_name in ["recall", "ndcg"]:
                    metrics_list[k][metric_name].append(metrics_dict[k][metric_name])
            best_recall, should_stop = early_stopping(metrics_list[k_min]["recall"], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]["recall"].index(best_recall) == len(epoch_list) - 1:
                save_jax_checkpoint(params, args.save_dir, epoch, best_epoch)
                logging.info("Save model on epoch %04d!", epoch)
                best_epoch = epoch

    metrics_df = [epoch_list]
    metrics_cols = ["epoch_idx"]
    for k in Ks:
        for metric_name in ["recall", "ndcg"]:
            metrics_df.append(metrics_list[k][metric_name])
            metrics_cols.append("{}@{}".format(metric_name, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + "/metrics.tsv", sep="\t", index=False)

    if best_epoch >= 0:
        best_metrics = metrics_df.loc[metrics_df["epoch_idx"] == best_epoch].iloc[0].to_dict()
        logging.info(
            "Best CF Evaluation: Epoch %04d | Recall [%.4f, %.4f], NDCG [%.4f, %.4f]",
            int(best_metrics["epoch_idx"]),
            best_metrics["recall@{}".format(k_min)],
            best_metrics["recall@{}".format(k_max)],
            best_metrics["ndcg@{}".format(k_min)],
            best_metrics["ndcg@{}".format(k_max)],
        )

    metadata_path = os.path.join(args.save_dir, "jax_run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "backend": args.backend,
                "precision": args.precision,
                "best_epoch": best_epoch,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    args = parse_t_akdn_args()
    train(args)
