"""
AKDN 勾配フロー可視化スクリプト

main_akdn.py と同じデータ読み込み・モデル構築を行い、
指定エポック数だけ学習した後の勾配フローを分析する。

使い方:
  python visualize_gradient_flow_akdn.py --data_name alibaba-fashion --n_warmup_epochs 0
  python visualize_gradient_flow_akdn.py --data_name alibaba-fashion --n_warmup_epochs 5 --use_pretrain 1
"""

import os
import sys
import random
import logging
from time import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.AKDN import AKDN
from parser.parser_akdn import parse_akdn_args
from utils.log_helper import *
from utils.model_helper import *
from data_loader.loader_akdn import DataLoaderAKDN


# ============================================================
# 1. パラメータ別 勾配ノルムテーブル
# ============================================================
def analyze_param_gradients(model, loss_value):
    """
    backward() 後の全パラメータの勾配状態を表形式で出力する。
    """
    print("\n" + "=" * 80)
    print("【1】パラメータ別 勾配ノルムテーブル")
    print("=" * 80)
    print(f"  Loss value: {loss_value:.6f}")
    print()

    header = f"{'Parameter':<45} {'Shape':<20} {'ReqGrad':<8} {'HasGrad':<8} {'GradNorm':<14} {'Status'}"
    print(header)
    print("-" * len(header))

    grad_info = []
    for name, param in model.named_parameters():
        req_grad = param.requires_grad
        has_grad = param.grad is not None
        grad_norm = param.grad.norm().item() if has_grad else 0.0
        
        if not req_grad:
            status = "❌ requires_grad=False"
        elif not has_grad:
            status = "❌ 勾配なし (計算グラフ切断)"
        elif grad_norm == 0.0:
            status = "⚠️  勾配ゼロ"
        elif grad_norm < 1e-10:
            status = "⚠️  勾配極小"
        else:
            status = "✅ 正常"
        
        shape_str = str(list(param.shape))
        print(f"  {name:<43} {shape_str:<20} {str(req_grad):<8} {str(has_grad):<8} {grad_norm:<14.8f} {status}")
        grad_info.append((name, req_grad, has_grad, grad_norm, status))
    
    # サマリー
    n_total = len(grad_info)
    n_no_grad = sum(1 for _, _, hg, _, _ in grad_info if not hg)
    n_zero = sum(1 for _, _, hg, gn, _ in grad_info if hg and gn == 0.0)
    n_ok = sum(1 for _, _, hg, gn, _ in grad_info if hg and gn > 0.0)
    
    print()
    print(f"  合計: {n_total} パラメータ | 正常: {n_ok} | 勾配なし: {n_no_grad} | 勾配ゼロ: {n_zero}")
    return grad_info


# ============================================================
# 2. 中間テンソルの勾配追跡
# ============================================================
def analyze_intermediate_gradients(model, data, device):
    """
    _compute_kg_attention 内部の各中間テンソルに retain_grad() を呼び、
    backward 後に勾配統計を出力する。
    
    モデル本体を変更せず、メソッドをモンキーパッチして計測する。
    """
    print("\n" + "=" * 80)
    print("【2】_compute_kg_attention 内部の中間テンソル勾配")
    print("=" * 80)

    # 中間テンソルを捕捉するための辞書
    intermediates = OrderedDict()

    # 元のメソッドを保存
    original_compute = model._compute_kg_attention
    call_counter = [0]

    def patched_compute_kg_attention(e_entities_curr):
        """
        _compute_kg_attention のモンキーパッチ版 (AKDN用)。
        TransR投影なし、semantic score のみ。
        """
        layer = call_counter[0]
        call_counter[0] += 1
        L = f'L{layer}'

        # 1. Embedding lookup
        h_embed = e_entities_curr[model.h_list]
        t_embed = e_entities_curr[model.t_list]
        r_embed = model.relation_embed(model.r_list)
        h_embed.retain_grad(); intermediates[f'h_embed ({L})'] = h_embed
        t_embed.retain_grad(); intermediates[f't_embed ({L})'] = t_embed
        r_embed.retain_grad(); intermediates[f'r_embed ({L})'] = r_embed

        # 2. Concatenate [neighbor, center]
        cat_embed = torch.cat([t_embed, h_embed], dim=1)
        cat_embed.retain_grad(); intermediates[f'cat_embed ({L})'] = cat_embed

        # 3. Linear Transform
        trans_embed = model.W_k(cat_embed)
        trans_embed.retain_grad(); intermediates[f'trans_embed ({L})'] = trans_embed

        # 4. Relation-aware Interaction
        product = trans_embed * r_embed
        product.retain_grad(); intermediates[f'product ({L})'] = product

        attention_logits = torch.sum(product, dim=1)
        attention_logits.retain_grad(); intermediates[f'attn_logits ({L})'] = attention_logits

        # 5. Activation
        attention_values = model.leakyrelu(attention_logits)
        attention_values.retain_grad(); intermediates[f'attn_values ({L})'] = attention_values

        # 6. Edge Softmax
        alpha = model._edge_softmax(attention_values)
        alpha.retain_grad(); intermediates[f'alpha ({L})'] = alpha

        return alpha

    # パッチ適用
    model._compute_kg_attention = patched_compute_kg_attention

    # forward + backward
    model.train()
    model.zero_grad()

    cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(
        data.train_user_dict, min(data.cf_batch_size, 1024)
    )
    cf_batch_user = cf_batch_user.to(device)
    cf_batch_pos_item = cf_batch_pos_item.to(device)
    cf_batch_neg_item = cf_batch_neg_item.to(device)

    loss = model.calc_loss(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
    loss.backward()

    # 結果表示
    print(f"\n  Loss: {loss.item():.6f}")
    print()
    
    header = f"  {'Tensor Name':<35} {'Shape':<20} {'HasGrad':<8} {'GradMean':<14} {'GradStd':<14} {'GradMin':<14} {'GradMax':<14} {'Status'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    grad_flow_broken_at = None
    prev_had_grad = True

    for name, tensor in intermediates.items():
        has_grad = tensor.grad is not None
        shape_str = str(list(tensor.shape))

        if has_grad:
            g = tensor.grad
            g_mean = g.mean().item()
            g_std = g.std().item()
            g_min = g.min().item()
            g_max = g.max().item()
            
            if g.abs().max().item() < 1e-12:
                status = "⚠️  実質ゼロ"
            else:
                status = "✅"
            
            print(f"  {name:<35} {shape_str:<20} {'True':<8} {g_mean:<14.8f} {g_std:<14.8f} {g_min:<14.8f} {g_max:<14.8f} {status}")
        else:
            status = "❌ 勾配なし"
            if prev_had_grad and grad_flow_broken_at is None:
                grad_flow_broken_at = name
                status = "❌ ★ここで切断★"
            print(f"  {name:<35} {shape_str:<20} {'False':<8} {'---':<14} {'---':<14} {'---':<14} {'---':<14} {status}")

        prev_had_grad = has_grad

    if grad_flow_broken_at:
        print(f"\n  ★ 勾配切断ポイント: {grad_flow_broken_at}")
    else:
        print(f"\n  ✅ 全中間テンソルに勾配が流れている")

    # メソッドを復元
    model._compute_kg_attention = original_compute

    return intermediates


# ============================================================
# 3. get_embeddings 全体の勾配フロー
# ============================================================
def analyze_get_embeddings_flow(model, data, device):
    """
    get_embeddings() 内の各ステップ（KG集約、IG集約、Fusion Gate）の
    出力テンソルに retain_grad() を設定し、勾配フローを追跡する。
    """
    print("\n" + "=" * 80)
    print("【3】get_embeddings ループ内の勾配フロー")
    print("=" * 80)

    intermediates = OrderedDict()
    original_get_embeddings = model.get_embeddings

    def patched_get_embeddings():
        all_embed = model.entity_user_embed.weight
        
        e_entities = all_embed[:model.n_entities]
        e_users = all_embed[model.n_entities:]
        
        e_entities.retain_grad(); intermediates['e_entities (init)'] = e_entities
        e_users.retain_grad(); intermediates['e_users (init)'] = e_users

        user_embeds_list = [e_users]
        item_collab_embeds_list = [e_entities]
        
        e_items_dual = e_entities
        e_users_curr = e_users
        e_entities_curr = e_entities

        if model.record_gate:
            model.gate_coefficients = []
            model.gate_inputs = []
            model.gate_wa_kg = []
            model.gate_wb_ig = []
            model.gate_ig = []
            model.gate_kg = []

        for i in range(model.n_layers):
            # KG Attention + Aggregation (最終層はスキップ: dead-end 回避)
            if i < model.n_layers - 1:
                alpha = model._compute_kg_attention(e_entities_curr)
                e_items_kg = model._kg_aggregation(alpha, e_entities_curr)
                e_items_kg.retain_grad()
                intermediates[f'e_items_kg (layer {i})'] = e_items_kg

            # IG Aggregation
            e_items_collab, e_users_new = model._ig_aggregation(e_items_dual, e_users_curr)
            e_items_collab.retain_grad(); intermediates[f'e_items_collab (layer {i})'] = e_items_collab
            e_users_new.retain_grad(); intermediates[f'e_users_new (layer {i})'] = e_users_new

            # Fusion Gate (最終層はIG出力をそのまま使用)
            if i < model.n_layers - 1:
                e_items_dual_new = model.fusion_gate(e_items_kg, e_items_collab)
            else:
                e_items_dual_new = e_items_collab
            e_items_dual_new.retain_grad()
            intermediates[f'e_items_fused (layer {i})'] = e_items_dual_new

            # Dropout
            if model.mess_dropout[i] > 0.0:
                e_items_collab = F.dropout(e_items_collab, p=model.mess_dropout[i], training=model.training)
                e_users_new = F.dropout(e_users_new, p=model.mess_dropout[i], training=model.training)
                e_items_dual_new = F.dropout(e_items_dual_new, p=model.mess_dropout[i], training=model.training)

            item_collab_embeds_list.append(e_items_collab)
            user_embeds_list.append(e_users_new)
            
            e_items_dual = e_items_dual_new
            e_users_curr = e_users_new
            if i < model.n_layers - 1:
                e_entities_curr = e_items_kg

        # 最終表現
        item_final = torch.stack(item_collab_embeds_list, dim=1).sum(dim=1)
        user_final = torch.stack(user_embeds_list, dim=1).sum(dim=1)
        
        item_final.retain_grad(); intermediates['item_final'] = item_final
        user_final.retain_grad(); intermediates['user_final'] = user_final

        return torch.cat([item_final, user_final], dim=0)

    model.get_embeddings = patched_get_embeddings

    # forward + backward
    model.train()
    model.zero_grad()

    cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(
        data.train_user_dict, min(data.cf_batch_size, 1024)
    )
    cf_batch_user = cf_batch_user.to(device)
    cf_batch_pos_item = cf_batch_pos_item.to(device)
    cf_batch_neg_item = cf_batch_neg_item.to(device)

    loss = model.calc_loss(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
    loss.backward()

    print(f"\n  Loss: {loss.item():.6f}")
    print()

    header = f"  {'Tensor Name':<35} {'Shape':<25} {'HasGrad':<8} {'GradNorm':<14} {'Status'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, tensor in intermediates.items():
        has_grad = tensor.grad is not None
        shape_str = str(list(tensor.shape))
        
        if has_grad:
            grad_norm = tensor.grad.norm().item()
            status = "✅" if grad_norm > 1e-10 else "⚠️  勾配極小"
            print(f"  {name:<35} {shape_str:<25} {'True':<8} {grad_norm:<14.8f} {status}")
        else:
            print(f"  {name:<35} {shape_str:<25} {'False':<8} {'---':<14} ❌ 勾配なし")

    model.get_embeddings = original_get_embeddings
    return intermediates


# ============================================================
# 4. torchviz 計算グラフ可視化 (オプション)
# ============================================================
def visualize_computation_graph(model, data, device, save_path='gradient_flow_graph_akdn'):
    """
    torchviz が利用可能であれば計算グラフを PDF で出力する。
    """
    print("\n" + "=" * 80)
    print("【4】計算グラフ可視化 (torchviz)")
    print("=" * 80)

    try:
        from torchviz import make_dot
    except ImportError:
        print("  torchviz が未インストール。スキップします。")
        print("  インストール: pip install torchviz")
        return

    model.train()
    model.zero_grad()

    cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(
        data.train_user_dict, min(data.cf_batch_size, 512)
    )
    cf_batch_user = cf_batch_user.to(device)
    cf_batch_pos_item = cf_batch_pos_item.to(device)
    cf_batch_neg_item = cf_batch_neg_item.to(device)

    loss = model.calc_loss(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)

    params = dict(model.named_parameters())
    dot = make_dot(loss, params=params, show_attrs=False, show_saved=False)
    dot.render(save_path, format='pdf', cleanup=True)
    print(f"  計算グラフを {save_path}.pdf に保存しました。")


# ============================================================
# メイン
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="AKDN Gradient Flow Visualization")

    # main_akdn.py と同じ引数
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--data_name', nargs='?', default='alibaba-fashion')
    parser.add_argument('--data_dir', nargs='?', default='datasets/')
    parser.add_argument('--use_pretrain', type=int, default=0)
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth')
    parser.add_argument('--cf_batch_size', type=int, default=4096)
    parser.add_argument('--test_batch_size', type=int, default=10000)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--relation_dim', type=int, default=64)
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 64, 64]')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]')
    parser.add_argument('--edge_dropout_rate', type=float, default=0.5)
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--Ks', nargs='?', default='[20]')

    # 勾配可視化専用の引数
    parser.add_argument('--n_warmup_epochs', type=int, default=0,
                        help='分析前に学習するウォームアップエポック数。0の場合は初期状態で分析。')
    parser.add_argument('--skip_torchviz', action='store_true',
                        help='torchviz による計算グラフ出力をスキップ')
    parser.add_argument('--graph_save_path', type=str, default='gradient_flow_graph_akdn',
                        help='torchviz 出力の保存パス (拡張子なし)')

    args = parser.parse_args()

    # save_dir (ログ出力には不要だがデータローダー互換性のため)
    args.save_dir = '/tmp/gradient_viz_akdn_logs/'
    args.n_epoch = args.n_warmup_epochs
    os.makedirs(args.save_dir, exist_ok=True)

    # シード固定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # デバイス
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # データ読み込み
    print("データ読み込み中...")
    data = DataLoaderAKDN(args, logging)

    # Pretrained Embeddings
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # モデル構築
    model = AKDN(args, data.n_users, data.n_entities, data.n_relations,
                 A_in=data.norm_adj_mat,
                 user_pre_embed=user_pre_embed,
                 item_pre_embed=item_pre_embed,
                 edge_dropout_rate=args.edge_dropout_rate)

    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)

    # KG構造セット
    relations = list(data.train_relation_dict.keys())
    model.set_kg_structure(
        data.h_list.to(device), data.t_list.to(device),
        data.r_list.to(device), relations
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ウォームアップ学習
    if args.n_warmup_epochs > 0:
        print(f"\n{'='*80}")
        print(f"ウォームアップ学習: {args.n_warmup_epochs} エポック")
        print(f"{'='*80}")
        
        n_batch = data.n_cf_train // data.cf_batch_size + 1
        
        for epoch in range(1, args.n_warmup_epochs + 1):
            model.train()
            total_loss = 0
            for iter_i in range(1, n_batch + 1):
                cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(
                    data.train_user_dict, data.cf_batch_size
                )
                cf_batch_user = cf_batch_user.to(device)
                cf_batch_pos_item = cf_batch_pos_item.to(device)
                cf_batch_neg_item = cf_batch_neg_item.to(device)

                batch_loss = model.calc_loss(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += batch_loss.item()

            print(f"  Epoch {epoch}/{args.n_warmup_epochs} | Mean Loss: {total_loss / n_batch:.4f}")

    # ============================================================
    # 勾配フロー分析の実行
    # ============================================================
    print(f"\n{'#'*80}")
    print(f"# AKDN 勾配フロー分析開始")
    print(f"# data_name={args.data_name}, warmup_epochs={args.n_warmup_epochs}")
    print(f"{'#'*80}")

    # 分析1: パラメータ別勾配ノルム
    model.train()
    model.zero_grad()

    cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(
        data.train_user_dict, min(data.cf_batch_size, 1024)
    )
    cf_batch_user = cf_batch_user.to(device)
    cf_batch_pos_item = cf_batch_pos_item.to(device)
    cf_batch_neg_item = cf_batch_neg_item.to(device)

    loss = model.calc_loss(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
    loss.backward()
    analyze_param_gradients(model, loss.item())

    # 分析2: 中間テンソルの勾配
    analyze_intermediate_gradients(model, data, device)

    # 分析3: get_embeddings ループ全体の勾配フロー
    analyze_get_embeddings_flow(model, data, device)

    # 分析4: 計算グラフ (オプション)
    if not args.skip_torchviz:
        visualize_computation_graph(model, data, device, save_path=args.graph_save_path)

    print(f"\n{'='*80}")
    print("分析完了!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
