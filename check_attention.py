import os
import sys
import random
import logging
from time import time

import numpy as np
import torch
import torch.optim as optim

from model.T_AKDN import T_AKDN
from parser.parser_t_akdn import parse_t_akdn_args
from utils.log_helper import logging_config, create_log_id
from data_loader.loader_akdn import DataLoaderAKDN

def check_attention(args):
    """
    CF(購買フロー)の学習・評価をすべてスキップし、
    KGE (Knowledge Graph Embedding) の学習だけを回して、
    毎エポックのアテンション診断結果を高速に出力するスクリプト。
    """
    # === 初期設定 ===
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}_attention_check'.format(log_save_id), no_console=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"--- Attention Diagnostics Check ---")
    logging.info(f"Device: {device}")
    logging.info(f"tau: {args.tau}, lambda_att: {args.lambda_att}, kge_lambda: {args.kge_lambda}")

    # === データ読み込み ===
    logging.info("DataLoader loading...")
    data = DataLoaderAKDN(args, logging)

    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # === モデル構築 ===
    model = T_AKDN(args, data.n_users, data.n_entities, data.n_relations, 
                   A_in=data.norm_adj_mat, 
                   user_pre_embed=user_pre_embed, 
                   item_pre_embed=item_pre_embed,
                   edge_dropout_rate=args.edge_dropout_rate)
    
    model.to(device)
    relations = list(data.train_relation_dict.keys())
    model.set_kg_structure(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device), relations)
    model.set_lambda(args.lambda_att)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # === KGE のみの学習ループ ===
    # CF のイテレーション数に合わせるか、KGのイテレーション数に合わせるか。
    # ここでは高速化のため、エポックあたりの回数を KG トリップレット数ベースで回す。
    n_kg_triplets = len(data.train_kg_dict)
    n_batch = max(n_kg_triplets // args.kg_batch_size, 1)

    logging.info(f"Start KGE-only training... (Epochs: {args.n_epoch}, Batches/Epoch: {n_batch})")

    for epoch in range(1, args.n_epoch + 1):
        model.train()
        total_kge_loss = 0
        t0 = time()

        for iter in range(1, n_batch + 1):
            kg_batch_h, kg_batch_r, kg_batch_pos_t, kg_batch_neg_t = data.generate_kg_batch(
                data.train_kg_dict, args.kg_batch_size, data.n_entities)
            
            kg_batch_h = kg_batch_h.to(device)
            kg_batch_r = kg_batch_r.to(device)
            kg_batch_pos_t = kg_batch_pos_t.to(device)
            kg_batch_neg_t = kg_batch_neg_t.to(device)

            kge_loss, kge_l2 = model('calc_kge_loss', kg_batch_h, kg_batch_r, kg_batch_pos_t, kg_batch_neg_t)
            
            # KGEのロスのみで勾配更新 (CFは無視)
            batch_loss = args.kge_lambda * kge_loss + args.kge_l2loss_lambda * kge_l2
            
            # もし kge_lambda == 0.0 の場合はロスがゼロになってしまうが、
            # Attentionの初期形状（ランダム初期化、またはPretrain）を確認するだけなら問題ない。
            if batch_loss.requires_grad and batch_loss.item() != 0.0:
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            total_kge_loss += kge_loss.item()

        # === 毎エポック アテンション診断を出力 ===
        model.eval()
        attn_diag = model.compute_attention_diagnostics(
            threshold=args.attn_diag_threshold, top_k=args.attn_diag_top_k)
        
        logging.info(f"Epoch {epoch:03d} [{time()-t0:.1f}s] | "
                     f"KGE Loss: {total_kge_loss/n_batch:.4f} | "
                     f"EffNeighbors: {attn_diag['effective_neighbors']:.2f} | "
                     f"Top{args.attn_diag_top_k} Ratio: {attn_diag['topk_ratio']:.4f}")

if __name__ == '__main__':
    # parse_t_akdn_args は main_t_akdn 用の全引数を含むのでそのまま流用
    args = parse_t_akdn_args()
    
    # KGE lossのみのテスト用なのでデフォルトのエポック数を減らす（コマンドライン引数で上書き可能）
    if '--n_epoch' not in sys.argv:
        args.n_epoch = 10
        
    check_attention(args)
