"""
Knowledge Graph 構造の静的分析スクリプト

データセットのKGをモデル非依存で分析し、
Attention学習に影響を与える構造的特徴を可視化する。

Usage:
    python analysis_knowledge_graph_attention.py --data_name yelp2018
"""

import os
import sys
import logging
import collections

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from parser.parser_akdn import parse_akdn_args
from data_loader.loader_akdn import DataLoaderAKDN


# =============================================================================
# 1. 基本統計
# =============================================================================
def analyze_basic_stats(data):
    """
    KGの基本統計: エンティティ数、リレーション数、トリプル数、アイテムKGカバー率。
    KGの規模感と、推薦対象アイテムがKGにどの程度含まれるかを把握する。
    """
    print("\n" + "=" * 60)
    print("1. 基本統計 (Basic Statistics)")
    print("=" * 60)

    kg = data.kg_train_data  # 逆関係追加済み
    n_triples = len(kg)
    n_entities = data.n_entities
    n_relations = data.n_relations
    n_items = data.n_items

    # ユニークな head / tail エンティティ
    unique_heads = kg['h'].nunique()
    unique_tails = kg['t'].nunique()
    unique_entities_in_kg = len(set(kg['h'].unique()) | set(kg['t'].unique()))

    # アイテムのKGカバー率 (アイテムIDは 0 ~ n_items-1)
    item_ids = set(range(n_items))
    kg_entity_ids = set(kg['h'].unique()) | set(kg['t'].unique())
    items_in_kg = item_ids & kg_entity_ids
    item_coverage = len(items_in_kg) / n_items if n_items > 0 else 0.0

    # 密度 (density = triples / (entities * entities * relations))
    density = n_triples / (n_entities * n_entities * n_relations) if n_entities > 0 else 0.0

    stats = {
        'n_entities (全エンティティ数)': n_entities,
        'n_relations (リレーション数, 逆関係含む)': n_relations,
        'n_triples (トリプル数, 逆関係含む)': n_triples,
        'n_triples (片方向のみ)': n_triples // 2,
        'KGに出現するユニークエンティティ数': unique_entities_in_kg,
        'ユニークHead数': unique_heads,
        'ユニークTail数': unique_tails,
        'n_items (アイテム数)': n_items,
        'KG中のアイテム数': len(items_in_kg),
        'アイテムKGカバー率': f'{item_coverage:.4f} ({item_coverage*100:.1f}%)',
        'グラフ密度': f'{density:.2e}',
    }

    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats


# =============================================================================
# 2. 次数分布
# =============================================================================
def analyze_degree_distribution(data):
    """
    エンティティの out/in/total 次数の統計量とヒストグラム。
    べき乗則の有無やハブノードの存在を確認する。
    """
    print("\n" + "=" * 60)
    print("2. 次数分布 (Degree Distribution)")
    print("=" * 60)

    kg = data.kg_train_data

    out_degree = collections.Counter(kg['h'].values)
    in_degree = collections.Counter(kg['t'].values)

    all_entities = set(out_degree.keys()) | set(in_degree.keys())
    total_degree = {e: out_degree.get(e, 0) + in_degree.get(e, 0) for e in all_entities}

    out_vals = np.array(list(out_degree.values()))
    in_vals = np.array(list(in_degree.values()))
    total_vals = np.array(list(total_degree.values()))

    for name, vals in [('Out-degree', out_vals), ('In-degree', in_vals), ('Total-degree', total_vals)]:
        print(f"\n  [{name}]")
        print(f"    Min: {np.min(vals)}, Max: {np.max(vals)}")
        print(f"    Mean: {np.mean(vals):.2f}, Median: {np.median(vals):.1f}")
        print(f"    Std: {np.std(vals):.2f}")
        percentiles = np.percentile(vals, [25, 50, 75, 90, 95, 99])
        print(f"    Percentiles [25, 50, 75, 90, 95, 99]: {percentiles}")

    return {
        'out_degree': out_vals,
        'in_degree': in_vals,
        'total_degree': total_vals,
        'out_degree_dict': out_degree,
        'in_degree_dict': in_degree,
        'total_degree_dict': total_degree,
    }


# =============================================================================
# 3. リレーション別統計
# =============================================================================
def analyze_relation_stats(data):
    """
    各リレーションタイプのトリプル数と関与エンティティ数。
    特定リレーションへの偏りがAttention学習に与えるリスクを評価する。
    """
    print("\n" + "=" * 60)
    print("3. リレーション別統計 (Relation Statistics)")
    print("=" * 60)

    kg = data.kg_train_data
    n_relations_original = data.n_relations // 2  # 逆関係を除いた数

    # リレーション名の読み込み試行
    relation_names = {}
    relation_file = os.path.join(data.data_dir, 'relation_list.txt')
    if os.path.exists(relation_file):
        with open(relation_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # ヘッダースキップ
                parts = line.strip().split()
                if len(parts) >= 2:
                    relation_names[int(parts[1])] = parts[0]

    print(f"\n  リレーション数 (片方向): {n_relations_original}")
    print(f"  リレーション数 (逆関係含む): {data.n_relations}")

    # リレーションごとの集計
    relation_groups = kg.groupby('r')
    rows = []
    for r_id, group in sorted(relation_groups, key=lambda x: len(x[1]), reverse=True):
        n_triples = len(group)
        n_heads = group['h'].nunique()
        n_tails = group['t'].nunique()
        is_inverse = r_id >= n_relations_original
        r_name = relation_names.get(r_id % n_relations_original, f'relation_{r_id}')
        label = f"{r_name} (inv)" if is_inverse else r_name
        rows.append({
            'r_id': r_id,
            'name': label,
            'n_triples': n_triples,
            'n_heads': n_heads,
            'n_tails': n_tails,
            'triples_ratio': n_triples / len(kg) * 100,
        })

    df = pd.DataFrame(rows)
    print(f"\n  {'ID':>4} {'Name':<45} {'Triples':>10} {'Ratio%':>7} {'Heads':>7} {'Tails':>7}")
    print("  " + "-" * 85)
    for _, row in df.iterrows():
        print(f"  {row['r_id']:>4} {row['name']:<45} {row['n_triples']:>10,} {row['triples_ratio']:>6.2f}% {row['n_heads']:>7,} {row['n_tails']:>7,}")

    return df


# =============================================================================
# 4. 近傍数の均一性 (Attention視点)
# =============================================================================
def analyze_neighborhood_uniformity(data):
    """
    ヘッドノードごとの近傍数分布。
    Softmax Attentionの実効性を評価する（近傍1→常にα=1.0、近傍数百→均一化）。
    """
    print("\n" + "=" * 60)
    print("4. 近傍数の均一性 (Neighborhood Uniformity for Attention)")
    print("=" * 60)

    kg_dict = data.train_kg_dict  # {head: [(tail, rel), ...]}
    neighbor_counts = np.array([len(neighbors) for neighbors in kg_dict.values()])

    print(f"\n  KG辞書に含まれるヘッドノード数: {len(kg_dict):,}")
    print(f"  近傍数の統計:")
    print(f"    Min: {np.min(neighbor_counts)}, Max: {np.max(neighbor_counts)}")
    print(f"    Mean: {np.mean(neighbor_counts):.2f}, Median: {np.median(neighbor_counts):.1f}")
    print(f"    Std: {np.std(neighbor_counts):.2f}")

    # Attention実効性の観点からの分類
    single_neighbor = np.sum(neighbor_counts == 1)
    few_neighbors = np.sum(neighbor_counts <= 3)
    moderate = np.sum((neighbor_counts >= 4) & (neighbor_counts <= 50))
    many = np.sum((neighbor_counts > 50) & (neighbor_counts <= 200))
    extreme = np.sum(neighbor_counts > 200)

    total = len(neighbor_counts)
    print(f"\n  Attention実効性の観点からの分類:")
    print(f"    近傍 = 1  (Attention無意味):   {single_neighbor:>8,} ({single_neighbor/total*100:>5.1f}%)")
    print(f"    近傍 ≤ 3  (Attention効果薄):   {few_neighbors:>8,} ({few_neighbors/total*100:>5.1f}%)")
    print(f"    近傍 4-50 (Attention効果的):    {moderate:>8,} ({moderate/total*100:>5.1f}%)")
    print(f"    近傍 51-200 (均一化リスク):     {many:>8,} ({many/total*100:>5.1f}%)")
    print(f"    近傍 > 200 (極端な均一化):      {extreme:>8,} ({extreme/total*100:>5.1f}%)")

    return neighbor_counts


# =============================================================================
# 5. アイテム vs 属性ノード
# =============================================================================
def analyze_item_vs_attribute(data):
    """
    アイテムノードと属性ノードの次数比較。
    属性ノードがハブとして機能しているかを確認する。
    """
    print("\n" + "=" * 60)
    print("5. アイテム vs 属性ノード (Item vs Attribute Nodes)")
    print("=" * 60)

    kg = data.kg_train_data
    n_items = data.n_items
    item_ids = set(range(n_items))

    # 全エンティティの次数
    out_degree = collections.Counter(kg['h'].values)
    in_degree = collections.Counter(kg['t'].values)
    total_degree = collections.Counter()
    total_degree.update(out_degree)
    total_degree.update(in_degree)

    # アイテム vs 属性 に分離
    item_degrees = []
    attr_degrees = []

    for entity_id, deg in total_degree.items():
        if entity_id in item_ids:
            item_degrees.append(deg)
        else:
            attr_degrees.append(deg)

    item_degrees = np.array(item_degrees) if item_degrees else np.array([0])
    attr_degrees = np.array(attr_degrees) if attr_degrees else np.array([0])

    # KGに存在しないアイテム
    items_not_in_kg = item_ids - set(total_degree.keys())

    print(f"\n  アイテムノード (KGに出現するもの): {len(item_degrees):,} / {n_items:,}")
    print(f"  属性ノード: {len(attr_degrees):,}")
    print(f"  KGに出現しないアイテム: {len(items_not_in_kg):,}")

    print(f"\n  [アイテムノードの次数]")
    print(f"    Mean: {np.mean(item_degrees):.2f}, Median: {np.median(item_degrees):.1f}")
    print(f"    Min: {np.min(item_degrees)}, Max: {np.max(item_degrees)}")

    print(f"\n  [属性ノードの次数]")
    print(f"    Mean: {np.mean(attr_degrees):.2f}, Median: {np.median(attr_degrees):.1f}")
    print(f"    Min: {np.min(attr_degrees)}, Max: {np.max(attr_degrees)}")

    # 属性ノードの中で高次数ハブを特定
    attr_degree_sorted = sorted(
        [(eid, deg) for eid, deg in total_degree.items() if eid not in item_ids],
        key=lambda x: x[1], reverse=True
    )
    print(f"\n  属性ノードの高次数ハブ Top-10:")
    for eid, deg in attr_degree_sorted[:10]:
        print(f"    Entity {eid}: degree = {deg:,}")

    return {
        'item_degrees': item_degrees,
        'attr_degrees': attr_degrees,
        'n_items_not_in_kg': len(items_not_in_kg),
    }


# =============================================================================
# 6. 連結成分分析
# =============================================================================
def analyze_connected_components(data):
    """
    Union-FindでKGの連結成分を分析。
    GNNの情報伝播が全ノードに届くかを確認する。
    """
    print("\n" + "=" * 60)
    print("6. 連結成分分析 (Connected Components)")
    print("=" * 60)

    kg = data.kg_train_data
    n_entities = data.n_entities

    # Union-Find
    parent = list(range(n_entities))
    rank = [0] * n_entities

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    # KGのエッジを使ってUnion
    for _, row in kg.iterrows():
        h, t = int(row['h']), int(row['t'])
        if h < n_entities and t < n_entities:
            union(h, t)

    # 連結成分の集計
    kg_entity_ids = set(kg['h'].unique()) | set(kg['t'].unique())
    component_sizes = collections.Counter()
    for eid in kg_entity_ids:
        if eid < n_entities:
            component_sizes[find(eid)] += 1

    n_components = len(component_sizes)
    sizes = sorted(component_sizes.values(), reverse=True)
    isolated = sum(1 for s in sizes if s == 1)

    # KGに出現しないエンティティ
    entities_not_in_kg = n_entities - len(kg_entity_ids)

    print(f"\n  連結成分数: {n_components:,}")
    print(f"  最大成分サイズ: {sizes[0]:,} ({sizes[0]/len(kg_entity_ids)*100:.1f}% of KG entities)" if sizes else "  N/A")
    if len(sizes) > 1:
        print(f"  2番目に大きい成分: {sizes[1]:,}")
    print(f"  孤立ノード (成分サイズ=1): {isolated:,}")
    print(f"  KGに出現しないエンティティ: {entities_not_in_kg:,}")

    if len(sizes) > 5:
        print(f"\n  成分サイズ Top-5: {sizes[:5]}")

    return {
        'n_components': n_components,
        'component_sizes': sizes,
        'isolated': isolated,
        'entities_not_in_kg': entities_not_in_kg,
    }


# =============================================================================
# 可視化
# =============================================================================
def plot_kg_analysis(degree_results, neighbor_counts, item_attr_results, save_path):
    """
    分析結果を1枚のPNGにまとめて可視化する。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Knowledge Graph Structure Analysis', fontsize=16, fontweight='bold')

    # --- (1) Total Degree Distribution (log-log) ---
    ax = axes[0, 0]
    total_deg = degree_results['total_degree']
    deg_counts = collections.Counter(total_deg)
    degs = sorted(deg_counts.keys())
    counts = [deg_counts[d] for d in degs]
    ax.scatter(degs, counts, s=5, alpha=0.6, color='#2196F3')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_title('(a) Total Degree Distribution (log-log)')
    ax.grid(True, alpha=0.3)

    # --- (2) Out-degree Distribution ---
    ax = axes[0, 1]
    ax.hist(degree_results['out_degree'], bins=100, color='#FF9800', alpha=0.7, log=True)
    ax.set_xlabel('Out-degree')
    ax.set_ylabel('Count (log)')
    ax.set_title('(b) Out-degree Distribution')
    ax.grid(True, alpha=0.3)

    # --- (3) In-degree Distribution ---
    ax = axes[0, 2]
    ax.hist(degree_results['in_degree'], bins=100, color='#4CAF50', alpha=0.7, log=True)
    ax.set_xlabel('In-degree')
    ax.set_ylabel('Count (log)')
    ax.set_title('(c) In-degree Distribution')
    ax.grid(True, alpha=0.3)

    # --- (4) Neighborhood Size Distribution ---
    ax = axes[1, 0]
    ax.hist(neighbor_counts, bins=100, color='#9C27B0', alpha=0.7, log=True)
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='=1 (Attention trivial)')
    ax.axvline(x=np.median(neighbor_counts), color='orange', linestyle='--', alpha=0.7,
               label=f'Median={np.median(neighbor_counts):.0f}')
    ax.set_xlabel('Number of Neighbors (per head node)')
    ax.set_ylabel('Count (log)')
    ax.set_title('(d) Neighborhood Size Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (5) Item vs Attribute Degree Comparison (Box Plot) ---
    ax = axes[1, 1]
    bp = ax.boxplot(
        [item_attr_results['item_degrees'], item_attr_results['attr_degrees']],
        labels=['Item Nodes', 'Attribute Nodes'],
        showfliers=False,  # 外れ値を表示しない（見やすさのため）
        patch_artist=True,
        boxprops=dict(facecolor='#E3F2FD'),
        medianprops=dict(color='red', linewidth=2)
    )
    ax.set_ylabel('Degree')
    ax.set_title('(e) Item vs Attribute Node Degree')
    ax.grid(True, alpha=0.3)

    # --- (6) Neighborhood Size CDF ---
    ax = axes[1, 2]
    sorted_nc = np.sort(neighbor_counts)
    cdf = np.arange(1, len(sorted_nc) + 1) / len(sorted_nc)
    ax.plot(sorted_nc, cdf, color='#E91E63', linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('Number of Neighbors (log)')
    ax.set_ylabel('CDF')
    ax.set_title('(f) Neighborhood Size CDF')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可視化を保存しました: {save_path}")


# =============================================================================
# メイン
# =============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    args = parse_akdn_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logging.info(f"Loading Data: {args.data_name}")
    data = DataLoaderAKDN(args, logging)

    print("\n" + "#" * 60)
    print(f"  Knowledge Graph Structure Analysis: {args.data_name}")
    print("#" * 60)

    # 1. 基本統計
    basic_stats = analyze_basic_stats(data)

    # 2. 次数分布
    degree_results = analyze_degree_distribution(data)

    # 3. リレーション別統計
    relation_df = analyze_relation_stats(data)

    # 4. 近傍数の均一性
    neighbor_counts = analyze_neighborhood_uniformity(data)

    # 5. アイテム vs 属性ノード
    item_attr_results = analyze_item_vs_attribute(data)

    # 6. 連結成分分析
    cc_results = analyze_connected_components(data)

    # 可視化
    save_path = os.path.join(args.save_dir, 'kg_structure_analysis.png')
    plot_kg_analysis(degree_results, neighbor_counts, item_attr_results, save_path)

    print("\n" + "=" * 60)
    print("分析完了")
    print("=" * 60)


if __name__ == '__main__':
    main()
