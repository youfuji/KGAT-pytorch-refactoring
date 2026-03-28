import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def analyze_dataset_characteristics(data, save_dir):
    """
    データセットの特性（ユーザー履歴長、アイテム人気度、KG情報量）を分析する
    """
    train_user_dict = data.train_user_dict
    test_user_dict = data.test_user_dict
    kg_dict = getattr(data, 'kg_dict', getattr(data, 'train_kg_dict', {}))
    
    # テスト対象のユーザーIDリスト
    test_users = list(test_user_dict.keys())
    
    # --- 1. User History Length (Sparsity) Analysis ---
    # テストユーザーが、学習データ時点で何件の履歴を持っていたか
    history_lengths = [len(train_user_dict[u]) if u in train_user_dict else 0 for u in test_users]
    
    # 統計量
    stats_sparsity = {
        'Min History': np.min(history_lengths),
        'Max History': np.max(history_lengths),
        'Mean History': np.mean(history_lengths),
        'Median History': np.median(history_lengths),
        'Ratio < 5': np.mean(np.array(history_lengths) < 5),
        'Ratio < 16 (Sparse)': np.mean(np.array(history_lengths) < 16),
        'Ratio >= 64 (Dense)': np.mean(np.array(history_lengths) >= 64)
    }
    
    # --- 2. Item Popularity Analysis (in Training Data) ---
    # 全アイテムの人気度（学習データでの出現回数）を計算
    all_interactions = []
    for items in train_user_dict.values():
        all_interactions.extend(items)
    item_counts = Counter(all_interactions)
    
    # 人気順にソートして、上位20%の閾値を決める（Headアイテム定義）
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
    total_interactions = sum(item_counts.values())
    current_sum = 0
    head_items = set()
    for item, count in sorted_items:
        current_sum += count
        head_items.add(item)
        if current_sum >= total_interactions * 0.2: # 20%ルール
            break
            
    # テストデータ（正解アイテム）の人気度をチェック
    test_items_popularity = []
    test_items_is_head = []
    
    for u in test_users:
        for item in test_user_dict[u]:
            pop = item_counts.get(item, 0)
            test_items_popularity.append(pop)
            test_items_is_head.append(1 if item in head_items else 0)
            
    stats_popularity = {
        'Mean Test Item Popularity': np.mean(test_items_popularity),
        'Head Item Ratio (in Test)': np.mean(test_items_is_head),
        'Tail Item Ratio (in Test)': 1.0 - np.mean(test_items_is_head)
    }

    # --- 3. KG Connectivity Analysis ---
    # テストアイテムがKG上でいくつ属性を持っているか
    kg_degrees = []
    has_kg_info = []
    
    for u in test_users:
        for item in test_user_dict[u]:
            if item in kg_dict:
                degree = len(kg_dict[item])
                kg_degrees.append(degree)
                has_kg_info.append(1)
            else:
                kg_degrees.append(0)
                has_kg_info.append(0)

    stats_kg = {
        'KG Coverage (Ratio of Test Items in KG)': np.mean(has_kg_info),
        'Mean KG Degree (All)': np.mean(kg_degrees),
        'Mean KG Degree (Existing Only)': np.mean([d for d in kg_degrees if d > 0]) if any(has_kg_info) else 0
    }

    # --- Summary Output ---
    print("\n=== Dataset Characteristics Analysis ===")
    print("1. User Sparsity:")
    for k, v in stats_sparsity.items():
        print(f"  {k}: {v:.4f}")
        
    print("\n2. Item Popularity (Test Data):")
    for k, v in stats_popularity.items():
        print(f"  {k}: {v:.4f}")
        
    print("\n3. KG Connectivity (Test Items):")
    for k, v in stats_kg.items():
        print(f"  {k}: {v:.4f}")

    # --- Plotting (Optional) ---
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(history_lengths, bins=50, color='skyblue', log=True)
    plt.title('User History Length (Log Scale)')
    plt.xlabel('Number of Interactions')
    plt.ylabel('User Count')
    
    plt.subplot(1, 3, 2)
    plt.hist(test_items_popularity, bins=50, color='salmon', log=True)
    plt.title('Test Item Popularity (Log Scale)')
    plt.xlabel('Item Popularity (Train Count)')
    
    plt.subplot(1, 3, 3)
    plt.hist([d for d in kg_degrees if d > 0], bins=30, color='lightgreen')
    plt.title('KG Degree Distribution (Test Items)')
    plt.xlabel('Number of Triplets')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/dataset_analysis.png')
    print(f"\nPlots saved to {save_dir}/dataset_analysis.png")

if __name__ == '__main__':
    import os
    import sys
    import logging
    
    # Add project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    
    from parser.parser_akdn import parse_akdn_args
    from data_loader.loader_akdn import DataLoaderAKDN

    # Logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    args = parse_akdn_args()
    
    # Ensure save directory exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    logging.info(f"Loading Data: {args.data_name}")
    data = DataLoaderAKDN(args, logging)
    
    analyze_dataset_characteristics(data, args.save_dir)