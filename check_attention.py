"""
check_attention.py — AKDN / T-AKDN 統一チェックポイント評価スクリプト

保存済み .pth をロードし、以下を評価・出力する:
  - Recall@K, NDCG@K  (CF推薦精度)
  - EffectiveNeighbors  (閾値超えの有効近傍数の平均)
  - Top-K Ratio         (上位K個のアテンション占有率の平均)

Usage:
  python check_attention.py \
    --model_type t_akdn \
    --checkpoint_path best_model/yelp2018/model_epoch490.pth \
    --data_name yelp2018
"""

import os
import sys
import argparse
import random
import logging
from time import time

import numpy as np
from tqdm import tqdm
import torch

from utils.log_helper import logging_config, create_log_id
from utils.model_helper import load_model
from utils.metrics import calc_metrics_at_k
from data_loader.loader_akdn import DataLoaderAKDN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved AKDN/T-AKDN checkpoint: Recall, NDCG, Attention Diagnostics.")

    # --- Required ---
    parser.add_argument('--model_type', type=str, required=True, choices=['akdn', 't_akdn'],
                        help='Model type: akdn or t_akdn')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the saved .pth checkpoint file.')

    # --- Data ---
    parser.add_argument('--data_name', type=str, default='yelp2018',
                        help='Dataset name (e.g. yelp2018, last-fm, amazon-book)')
    parser.add_argument('--data_dir', type=str, default='datasets/',
                        help='Input data path.')

    # --- Model architecture (must match the checkpoint) ---
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--relation_dim', type=int, default=64)
    parser.add_argument('--conv_dim_list', type=str, default='[64, 64, 64]')
    parser.add_argument('--mess_dropout', type=str, default='[0.1, 0.1, 0.1]')
    parser.add_argument('--edge_dropout_rate', type=float, default=0.5)
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5)
    parser.add_argument('--use_pretrain', type=int, default=0)
    parser.add_argument('--pretrain_embedding_dir', type=str, default='datasets/pretrain/')

    # --- T-AKDN specific (ignored for AKDN) ---
    parser.add_argument('--transr_dim', type=int, default=64)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--lambda_att', type=float, default=0.5)
    parser.add_argument('--kge_lambda', type=float, default=0.1)
    parser.add_argument('--kge_l2loss_lambda', type=float, default=1e-5)

    # --- Evaluation ---
    parser.add_argument('--Ks', type=str, default='[20]',
                        help='K values for Recall@K, NDCG@K (e.g. "[20, 40]")')
    parser.add_argument('--test_batch_size', type=int, default=10000)

    # --- Attention Diagnostics ---
    parser.add_argument('--attn_diag_threshold', type=float, default=0.05,
                        help='Threshold for effective neighborhood size.')
    parser.add_argument('--attn_diag_top_k', type=int, default=5,
                        help='Top-K neighbors for attention ratio.')

    args = parser.parse_args()

    # save_dir for logging
    args.save_dir = os.path.dirname(args.checkpoint_path) or '.'

    return args


def build_model(args, data, device):
    """model_type に応じて AKDN or T_AKDN を構築し、KG構造をセットして返す。"""
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    if args.model_type == 'akdn':
        from model.AKDN import AKDN
        model = AKDN(args, data.n_users, data.n_entities, data.n_relations,
                     A_in=data.norm_adj_mat,
                     user_pre_embed=user_pre_embed,
                     item_pre_embed=item_pre_embed,
                     edge_dropout_rate=args.edge_dropout_rate)
    else:
        from model.T_AKDN import T_AKDN
        model = T_AKDN(args, data.n_users, data.n_entities, data.n_relations,
                       A_in=data.norm_adj_mat,
                       user_pre_embed=user_pre_embed,
                       item_pre_embed=item_pre_embed,
                       edge_dropout_rate=args.edge_dropout_rate)

    model.to(device)
    relations = list(data.train_relation_dict.keys())
    model.set_kg_structure(data.h_list.to(device), data.t_list.to(device),
                           data.r_list.to(device), relations)

    if args.model_type == 't_akdn':
        model.set_lambda(args.lambda_att)

    return model


def evaluate(model, dataloader, Ks, device):
    """Recall@K, NDCG@K を計算して返す。"""
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size]
                        for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    metric_names = ['recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model('calc_score', batch_user_ids, item_ids)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(
                batch_scores, train_user_dict, test_user_dict,
                batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)

            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()

    return metrics_dict


def main():
    args = parse_args()

    # --- Seed ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- Logging ---
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir,
                   name='log{:d}_checkpoint_eval'.format(log_save_id),
                   no_console=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("=" * 60)
    logging.info("Checkpoint Evaluation")
    logging.info("=" * 60)
    logging.info(f"Model:      {args.model_type.upper()}")
    logging.info(f"Checkpoint: {args.checkpoint_path}")
    logging.info(f"Dataset:    {args.data_name}")
    logging.info(f"Device:     {device}")
    if args.model_type == 't_akdn':
        logging.info(f"tau: {args.tau}, lambda_att: {args.lambda_att}")

    # --- Data ---
    logging.info("Loading data...")
    # DataLoaderAKDN expects args.test_batch_size
    data = DataLoaderAKDN(args, logging)
    Ks = eval(args.Ks)

    # --- Model ---
    logging.info("Building model...")
    model = build_model(args, data, device)
    model = load_model(model, args.checkpoint_path)
    model.to(device)
    logging.info(f"Loaded checkpoint: {args.checkpoint_path}")

    # --- CF Evaluation (Recall, NDCG) ---
    logging.info("Running CF evaluation...")
    t0 = time()
    metrics_dict = evaluate(model, data, Ks, device)
    eval_time = time() - t0

    # --- Attention Diagnostics ---
    logging.info("Running attention diagnostics...")
    attn_diag = model.compute_attention_diagnostics(
        threshold=args.attn_diag_threshold, top_k=args.attn_diag_top_k)

    # --- Results ---
    logging.info("=" * 60)
    logging.info("Results")
    logging.info("=" * 60)

    for k in Ks:
        recall = metrics_dict[k]['recall']
        ndcg = metrics_dict[k]['ndcg']
        logging.info(f"  Recall@{k}: {recall:.4f}  |  NDCG@{k}: {ndcg:.4f}")

    logging.info(f"  EffectiveNeighbors (threshold={args.attn_diag_threshold}): "
                 f"{attn_diag['effective_neighbors']:.2f}")
    logging.info(f"  Top{args.attn_diag_top_k} Ratio: {attn_diag['topk_ratio']:.4f}")
    logging.info(f"  Evaluation time: {eval_time:.1f}s")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
