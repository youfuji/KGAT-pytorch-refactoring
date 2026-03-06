import os
import sys
import random
from time import time
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from model.T_AKDN import T_AKDN
from parser.parser_t_akdn import parse_t_akdn_args
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_akdn import DataLoaderAKDN


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model('calc_score', batch_user_ids, item_ids)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)

            cf_scores.append(batch_scores.numpy())
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
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data (DataLoaderAKDN をそのまま再利用)
    data = DataLoaderAKDN(args, logging)
    
    # Pretrained Embeddings (if available)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = T_AKDN(args, data.n_users, data.n_entities, data.n_relations, 
                   A_in=data.norm_adj_mat, 
                   user_pre_embed=user_pre_embed, 
                   item_pre_embed=item_pre_embed,
                   edge_dropout_rate=args.edge_dropout_rate)
                 
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    # グラフ構造をセット (GPU転送後)
    relations = list(data.train_relation_dict.keys())
    model.set_kg_structure(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device), relations)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    loss_metrics_list = {'cf_loss': [], 'kge_loss': []}
    attn_metrics_list = {'eff_neighbors': [], 'topk_ratio': []}
    metrics_list = {k: {'recall': [], 'ndcg': []} for k in Ks}

    # train model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # 3-phase Lambda annealing
        # Phase 1: warmup (λ=init) → Phase 2: linear anneal → Phase 3: saturation (λ=final)
        if epoch <= args.lambda_warmup_epochs:
            lam_val = args.lambda_init
        elif epoch <= args.lambda_warmup_epochs + args.lambda_anneal_epochs:
            progress = (epoch - args.lambda_warmup_epochs) / args.lambda_anneal_epochs
            lam_val = args.lambda_init + (args.lambda_final - args.lambda_init) * progress
        else:
            lam_val = args.lambda_final
        model.set_lambda(lam_val)

        time_cf = time()
        total_loss = 0
        total_cf_loss = 0
        total_kge_loss = 0
        n_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_batch + 1):
            time_iter = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            # ★ KG batch sampling for KGE multi-task loss
            kg_batch_h, kg_batch_r, kg_batch_pos_t, kg_batch_neg_t = data.generate_kg_batch(
                data.train_kg_dict, args.kg_batch_size, data.n_entities)
            kg_batch_h = kg_batch_h.to(device)
            kg_batch_r = kg_batch_r.to(device)
            kg_batch_pos_t = kg_batch_pos_t.to(device)
            kg_batch_neg_t = kg_batch_neg_t.to(device)

            # CF Loss (BPR + L2)
            cf_loss = model('calc_loss', cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)

            # ★ KGE Pairwise Ranking Loss
            kge_loss, kge_l2 = model('calc_kge_loss', kg_batch_h, kg_batch_r, kg_batch_pos_t, kg_batch_neg_t)

            # Total Loss = CF + λ_KGE * KGE + λ_KGE_L2 * KGE_L2
            batch_loss = cf_loss + args.kge_lambda * kge_loss + args.kge_l2loss_lambda * kge_l2

            if np.isnan(batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit()

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()
            total_cf_loss += cf_loss.item()
            total_kge_loss += kge_loss.item()

            if (iter % args.cf_print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Total {:.4f} | CF {:.4f} | KGE {:.4f} | Lambda {:.4f}'.format(
                    epoch, iter, n_batch, time() - time_iter, batch_loss.item(), cf_loss.item(), kge_loss.item(), lam_val))
        
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Mean Loss {:.4f} | Lambda {:.4f}'.format(
            epoch, n_batch, time() - time_cf, total_loss / n_batch, lam_val))
        logging.info('--- Group 1 (Loss) ---  BPR: {:.4f} | KGE: {:.4f}'.format(
            total_cf_loss / n_batch, total_kge_loss / n_batch))
        logging.info('Epoch {:04d} finished | Total Time {:.1f}s'.format(epoch, time() - time0))

        # Evaluate
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time_eval = time()
            _, metrics_dict = evaluate(model, data, Ks, device)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, time() - time_eval, metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

            # --- Group 2: Attention Diagnostics ---
            attn_diag = model.compute_attention_diagnostics(
                threshold=args.attn_diag_threshold, top_k=args.attn_diag_top_k)
            logging.info('--- Group 2 (Attention) ---  EffNeighbors: {:.2f} | Top{:d} Ratio: {:.4f}'.format(
                attn_diag['effective_neighbors'], args.attn_diag_top_k, attn_diag['topk_ratio']))

            epoch_list.append(epoch)
            loss_metrics_list['cf_loss'].append(total_cf_loss / n_batch)
            loss_metrics_list['kge_loss'].append(total_kge_loss / n_batch)
            attn_metrics_list['eff_neighbors'].append(attn_diag['effective_neighbors'])
            attn_metrics_list['topk_ratio'].append(attn_diag['topk_ratio'])
            
            for k in Ks:
                for m in ['recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # save metrics
    metrics_df = [
        epoch_list,
        loss_metrics_list['cf_loss'],
        loss_metrics_list['kge_loss'],
        attn_metrics_list['eff_neighbors'],
        attn_metrics_list['topk_ratio']
    ]
    metrics_cols = ['epoch_idx', 'cf_loss', 'kge_loss', 'eff_neighbors', 'topk_ratio']
    for k in Ks:
        for m in ['recall', 'ndcg']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info('Best CF Evaluation: Epoch {:04d} | Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        int(best_metrics['epoch_idx']), best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)]))


def predict(args):
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderAKDN(args, logging)

    # load model
    model = T_AKDN(args, data.n_users, data.n_entities, data.n_relations, A_in=data.norm_adj_mat)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    cf_scores, metrics_dict = evaluate(model, data, Ks, device)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))


if __name__ == '__main__':
    args = parse_t_akdn_args()
    train(args)
    # predict(args)
