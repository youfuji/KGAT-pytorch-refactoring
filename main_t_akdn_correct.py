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

from model.T_AKDN_correct import T_AKDN_correct
from parser.parser_t_akdn_correct import parse_t_akdn_correct_args
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_t_akdn_correct import DataLoaderTAKDNCorrect


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
                # mode='calc_score' (AKDN.pyの実装に合わせる)
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

    # Set up directory for this run
    log_save_id = 0
    while os.path.exists(os.path.join(args.save_dir, 'log{:d}'.format(log_save_id))):
        log_save_id += 1
    args.save_dir = os.path.join(args.save_dir, 'log{:d}/'.format(log_save_id))
    
    logging_config(folder=args.save_dir, name='log', no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderTAKDNCorrect(args, logging)
    
    # Pretrained Embeddings (if available)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    # T_AKDN_correctには IG用の隣接行列(norm_adj_mat) を渡す
    model = T_AKDN_correct(args, data.n_users, data.n_items, data.n_entities, data.n_relations, 
                 ig_adj_user_to_item=data.norm_adj_user_to_item,
                 ig_adj_item_to_user=data.norm_adj_item_to_user,
                 user_pre_embed=user_pre_embed, 
                 item_pre_embed=item_pre_embed,
                 edge_dropout_rate=args.edge_dropout_rate)
                 
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    # T_AKDN_correctのAttention計算用にグラフ構造をセットする (GPU転送後)
    relations = list(data.train_relation_dict.keys())
    model.set_kg_structure(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device), relations)

    # T_AKDN_correctはエンドツーエンド学習なので単一のOptimizerを使用
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'recall': [], 'ndcg': []} for k in Ks}

    # train model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # ------------------------------------------------------------------
        # 1. Train CF (Main Task) with Differentiable Attention
        # ------------------------------------------------------------------
        time_cf = time()
        total_loss = 0
        total_g_lambda = 0
        total_layer_g_lambda = None
        n_batch = data.n_cf_train // data.cf_batch_size + 1

        _mem_profiled = False
        for iter in range(1, n_batch + 1):
            time_iter = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            _do_mem = (not _mem_profiled) and torch.cuda.is_available()

            # T_AKDN_correct の calc_loss を呼び出し
            batch_loss, batch_g_lambda, batch_layer_g_lambda = model('calc_loss', cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
            if _do_mem:
                print(f"  [MEM] after forward (calc_loss): allocated={torch.cuda.memory_allocated()/1024**2:.1f}MB, reserved={torch.cuda.memory_reserved()/1024**2:.1f}MB")

            if np.isnan(batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit()

            batch_loss.backward()
            if _do_mem:
                print(f"  [MEM] after backward: allocated={torch.cuda.memory_allocated()/1024**2:.1f}MB, reserved={torch.cuda.memory_reserved()/1024**2:.1f}MB")

            optimizer.step()
            optimizer.zero_grad()
            if _do_mem:
                print(f"  [MEM] after optimizer.step + zero_grad: allocated={torch.cuda.memory_allocated()/1024**2:.1f}MB, reserved={torch.cuda.memory_reserved()/1024**2:.1f}MB")
                _mem_profiled = True

            total_loss += batch_loss.item()
            total_g_lambda += batch_g_lambda.item()
            batch_layer_g_lambda_cpu = batch_layer_g_lambda.detach().cpu()
            if total_layer_g_lambda is None:
                total_layer_g_lambda = torch.zeros_like(batch_layer_g_lambda_cpu)
            total_layer_g_lambda += batch_layer_g_lambda_cpu

            if (iter % args.cf_print_every) == 0:
                layer_lambda_text = ','.join(['L{}:{:.4f}'.format(i + 1, v) for i, v in enumerate(batch_layer_g_lambda_cpu.tolist())])
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f} | g_lambda {:.4f} | g_lambda Mean {:.4f} | layer_g_lambda [{}]'.format(epoch, iter, n_batch, time() - time_iter, batch_loss.item(), total_loss / iter, batch_g_lambda.item(), total_g_lambda / iter, layer_lambda_text))
        
        if total_layer_g_lambda is None:
            epoch_layer_lambda_text = ''
        else:
            epoch_layer_lambda = total_layer_g_lambda / n_batch
            epoch_layer_lambda_text = ','.join(['L{}:{:.4f}'.format(i + 1, v) for i, v in enumerate(epoch_layer_lambda.tolist())])
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f} | g_lambda Mean {:.4f} | layer_g_lambda Mean [{}]'.format(epoch, n_batch, time() - time_cf, total_loss / n_batch, total_g_lambda / n_batch, epoch_layer_lambda_text))
        logging.info('Epoch {:04d} finished | Total Time {:.1f}s'.format(epoch, time() - time0))

        # ------------------------------------------------------------------
        # 3. Evaluate
        # ------------------------------------------------------------------
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time_eval = time()
            _, metrics_dict = evaluate(model, data, Ks, device)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, time() - time_eval, metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

            epoch_list.append(epoch)
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
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['recall', 'ndcg']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(os.path.join(args.save_dir, 'metrics.tsv'), sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info('Best CF Evaluation: Epoch {:04d} | Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        int(best_metrics['epoch_idx']), best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)]))


def predict(args):
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderTAKDNCorrect(args, logging)

    # load model
    model = T_AKDN_correct(args, data.n_users, data.n_items, data.n_entities, data.n_relations, 
                 ig_adj_user_to_item=data.norm_adj_user_to_item,
                 ig_adj_item_to_user=data.norm_adj_item_to_user)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    cf_scores, metrics_dict = evaluate(model, data, Ks, device)
    np.save(os.path.join(args.save_dir, 'cf_scores.npy'), cf_scores)
    print('CF Evaluation: Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))


if __name__ == '__main__':
    args = parse_t_akdn_correct_args()
    train(args)
    # predict(args)
