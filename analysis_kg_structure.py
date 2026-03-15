"""
KG構造分析スクリプト
KGの次数分布をアイテム/属性に分けて分析し、Excelファイルに出力する。
"""
import os
import sys
import logging
import collections

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from parser.parser_akdn import parse_akdn_args
from data_loader.loader_akdn import DataLoaderAKDN


def analyze_kg_structure(data):
    """
    train_kg_dictから次数分布を計算し、アイテムと属性に分けて統計量を表示・Excel出力する。
    """
    kg_dict = data.train_kg_dict
    n_items = data.n_items

    # --- 次数の計算 ---
    item_degrees = []
    attribute_degrees = []

    for entity_id, neighbors in kg_dict.items():
        degree = len(neighbors)
        if entity_id < n_items:
            item_degrees.append(degree)
        else:
            attribute_degrees.append(degree)

    item_degrees = np.array(item_degrees)
    attribute_degrees = np.array(attribute_degrees)
    
    # ユーザーの次数計算 (data.train_user_dict から取得)
    user_degrees = []
    for user_id, items in data.train_user_dict.items():
        user_degrees.append(len(items))
    user_degrees = np.array(user_degrees)

    # --- サマリー統計量の表示 ---
    def print_stats(name, degrees):
        print(f"\n=== {name} ===")
        print(f"  Total entities: {len(degrees)}")
        print(f"  Total edges:    {int(np.sum(degrees))}")
        print(f"  Mean degree:    {np.mean(degrees):.2f}")
        print(f"  Median degree:  {np.median(degrees):.2f}")
        print(f"  Std degree:     {np.std(degrees):.2f}")
        print(f"  Min degree:     {int(np.min(degrees))}")
        print(f"  Max degree:     {int(np.max(degrees))}")

    print_stats("Items (ID < n_items)", item_degrees)
    print_stats("Attributes (ID >= n_items)", attribute_degrees)
    print_stats("Users", user_degrees)

    # --- Excel出力（アイテムと属性でシートを分ける） ---
    def degree_distribution_df(degrees):
        counter = collections.Counter(degrees.tolist())
        df = pd.DataFrame(
            sorted(counter.items()),
            columns=["degree", "count"]
        )
        return df

    item_df = degree_distribution_df(item_degrees)
    attr_df = degree_distribution_df(attribute_degrees)
    user_df = degree_distribution_df(user_degrees)

    output_path = os.path.join(data.args.save_dir, "kg_degree_distribution.xlsx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        item_df.to_excel(writer, sheet_name="items", index=False)
        attr_df.to_excel(writer, sheet_name="attributes", index=False)
        user_df.to_excel(writer, sheet_name="users", index=False)

    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    args = parse_akdn_args()
    logging.info(f"Loading Data: {args.data_name}")

    data = DataLoaderAKDN(args, logging)
    analyze_kg_structure(data)
