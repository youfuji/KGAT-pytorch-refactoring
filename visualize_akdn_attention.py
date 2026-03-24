import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_loader.loader_akdn import DataLoaderAKDN
from model.AKDN import AKDN
from parser.parser_akdn import parse_akdn_args
from utils.metrics import compute_attention_diagnostics_from_alpha
from utils.model_helper import load_model

# python visualize_akdn_attention.py \
#   --data_name yelp2018 \
#   --pretrain_model_path trained_model/.../model_epochXXX.pth \
#   --item_ids 10,25 \
#   --top_k 10


def parse_attention_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--focus_item', type=int, default=None)
    parser.add_argument('--item_ids', type=str, default=None,
                        help='Comma separated item ids for per-item plots.')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--layer_ids', type=str, default=None,
                        help='Comma separated layer ids to plot. Default: all recorded layers.')

    local_args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    base_args = parse_akdn_args()

    if local_args.save_dir:
        base_args.save_dir = local_args.save_dir

    return base_args, local_args


def parse_optional_int_list(raw_value):
    if raw_value is None or raw_value == '':
        return []
    return [int(token.strip()) for token in raw_value.split(',') if token.strip() != '']


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def summarize_tensor(values):
    if values.numel() == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
        }
    values_np = values.numpy()
    return {
        'mean': float(np.mean(values_np)),
        'std': float(np.std(values_np)),
        'min': float(np.min(values_np)),
        'max': float(np.max(values_np)),
    }


def plot_histogram(values, title, xlabel, save_path, color):
    if values.numel() == 0:
        return
    values_np = values.numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(values_np, bins=50, alpha=0.8, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    stats = summarize_tensor(values)
    plt.text(
        0.05,
        0.95,
        'mean={:.4f}\nstd={:.4f}\nmin={:.4f}\nmax={:.4f}'.format(
            stats['mean'], stats['std'], stats['min'], stats['max']
        ),
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_item_edges(record, item_id, top_k, save_path):
    item_mask = record['h'] == item_id
    if int(item_mask.sum().item()) == 0:
        return False

    df = pd.DataFrame({
        'tail': record['t'][item_mask].numpy(),
        'relation': record['r'][item_mask].numpy(),
        'attention_value': record['attention_value'][item_mask].numpy(),
        'alpha': record['alpha'][item_mask].numpy(),
    }).sort_values('alpha', ascending=False).head(top_k)

    labels = ['t{}|r{}'.format(int(t), int(r)) for t, r in zip(df['tail'], df['relation'])]
    x = np.arange(len(df))
    width = 0.28

    plt.figure(figsize=(max(8, len(df) * 1.4), 6))
    plt.bar(x - 0.5 * width, df['attention_value'], width=width, label='attention_value', color='tab:blue')
    plt.bar(x + 0.5 * width, df['alpha'], width=width, label='alpha', color='tab:red')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.title('Layer {}: item {} top-{} edges'.format(record['layer'] + 1, item_id, len(df)))
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return True


def export_layer_tables(record, save_dir, top_k):
    layer_df = pd.DataFrame({
        'head_item': record['h'].numpy(),
        'tail_entity': record['t'].numpy(),
        'relation': record['r'].numpy(),
        'attention_value': record['attention_value'].numpy(),
        'alpha': record['alpha'].numpy(),
    })
    layer_df.to_csv(os.path.join(save_dir, 'layer_{}_edges.csv'.format(record['layer'] + 1)), index=False)
    layer_df.sort_values('alpha', ascending=False).head(top_k).to_csv(
        os.path.join(save_dir, 'layer_{}_top_alpha_edges.csv'.format(record['layer'] + 1)),
        index=False,
    )


def collect_focus_items(records, local_args):
    focus_items = parse_optional_int_list(local_args.item_ids)
    if local_args.focus_item is not None:
        focus_items.append(local_args.focus_item)

    if focus_items:
        return sorted(set(focus_items))

    auto_items = []
    for record in records:
        if record['alpha'].numel() == 0:
            continue
        top_index = int(torch.argmax(record['alpha']).item())
        auto_items.append(int(record['h'][top_index].item()))
    return sorted(set(auto_items))


def filter_records_by_layer(records, local_args):
    requested_layers = parse_optional_int_list(local_args.layer_ids)
    if not requested_layers:
        return records

    requested = set(layer_id - 1 for layer_id in requested_layers)
    return [record for record in records if record['layer'] in requested]


def compute_layer_attention_diagnostics(record, n_items, threshold, top_k):
    diagnostics = compute_attention_diagnostics_from_alpha(
        alpha=record['alpha'],
        h_list=record['h'],
        n_items=n_items,
        threshold=threshold,
        top_k=top_k,
    )
    diagnostics.update({
        'layer': record['layer'] + 1,
        'threshold': threshold,
        'top_k': top_k,
    })
    return diagnostics


def main():
    args, local_args = parse_attention_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Loading data...')
    data = DataLoaderAKDN(args, logging)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    logging.info('Initializing model...')
    model = AKDN(
        args,
        data.n_users,
        data.n_items,
        data.n_entities,
        data.n_relations,
        A_in=data.norm_adj_mat,
        user_pre_embed=user_pre_embed,
        item_pre_embed=item_pre_embed,
        edge_dropout_rate=0.0,
    )

    if args.pretrain_model_path and os.path.exists(args.pretrain_model_path):
        logging.info('Loading model from %s', args.pretrain_model_path)
        model = load_model(model, args.pretrain_model_path)
    else:
        logging.warning('No pretrain_model_path specified or file not found. Using initialized weights.')

    model.to(device)
    model.eval()

    relations = list(data.train_relation_dict.keys())
    model.set_kg_structure(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device), relations)

    logging.info('Recording attention tensors from a single evaluation forward pass...')
    model.record_attention = True
    with torch.no_grad():
        model.get_embeddings()

    records = filter_records_by_layer(model.attention_records, local_args)
    if not records:
        raise RuntimeError('No attention records were collected for the requested layers.')

    attention_dir = os.path.join(args.save_dir, 'attention_viz')
    ensure_dir(attention_dir)

    summary_rows = []
    diagnostics_rows = []
    focus_items = collect_focus_items(records, local_args)
    logging.info('Focus items: %s', focus_items)

    colors = {
        'attention_value': 'tab:blue',
        'alpha': 'tab:red',
    }

    for record in records:
        layer_dir = os.path.join(attention_dir, 'layer_{}'.format(record['layer'] + 1))
        ensure_dir(layer_dir)

        diagnostics = compute_layer_attention_diagnostics(
            record,
            n_items=data.n_items,
            threshold=args.attn_diag_threshold,
            top_k=args.attn_diag_top_k,
        )
        diagnostics_rows.append(diagnostics)
        pd.DataFrame([diagnostics]).to_csv(
            os.path.join(layer_dir, 'attention_diagnostics.csv'),
            index=False,
        )
        logging.info(
            'Layer %d diagnostics | Effective Neighbors (> %.2f): %.4f | Top-%d Attention Ratio: %.4f',
            diagnostics['layer'],
            diagnostics['threshold'],
            diagnostics['effective_neighbors'],
            diagnostics['top_k'],
            diagnostics['topk_ratio'],
        )

        for key, color in colors.items():
            stats = summarize_tensor(record[key])
            stats.update({'layer': record['layer'] + 1, 'metric': key})
            summary_rows.append(stats)
            plot_histogram(
                record[key],
                'Layer {}: {}'.format(record['layer'] + 1, key),
                key,
                os.path.join(layer_dir, '{}_hist.png'.format(key)),
                color,
            )

        export_layer_tables(record, layer_dir, local_args.top_k)

        for item_id in focus_items:
            plotted = plot_item_edges(
                record,
                item_id=item_id,
                top_k=local_args.top_k,
                save_path=os.path.join(layer_dir, 'item_{}_top_edges.png'.format(item_id)),
            )
            if not plotted:
                logging.info('Layer %d has no item-edge records for item %d', record['layer'] + 1, item_id)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(attention_dir, 'attention_summary.csv'), index=False)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_df.to_csv(os.path.join(attention_dir, 'attention_diagnostics_summary.csv'), index=False)
    logging.info('Saved attention visualizations to %s', attention_dir)


if __name__ == '__main__':
    main()
