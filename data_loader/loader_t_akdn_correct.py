import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_loader.loader_base import DataLoaderBase


class DataLoaderTAKDNCorrect(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)

        # AKDN: IG（Interaction Graph）のための正規化隣接行列を作成
        self.create_ig_adjacency()

    def construct_data(self, kg_data):
        # 1. KGの構築 (TransR & Attention用)
        # 逆関係（Inverse Relations）の追加: (h, r, t) -> (t, r+n, h)
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        # IDシフトの廃止: User IDは純粋に 0 ~ n_users-1 として扱う
        # cf_train_data などのシフト(+self.n_entities)を削除しました。

        # [変更点]: KGATとは異なり、ここでCFデータをKGデータに統合しません。
        # AKDNではKGとIGを分離して扱うため、kg_train_dataは純粋なKnowledge Graphのみとします。
        self.kg_train_data = kg_data
        self.n_kg_train = len(self.kg_train_data)

        # KG辞書の構築 (Attention機構での近傍探索用)
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse_coo_tensor(i, v, torch.Size(shape))

    def create_ig_adjacency(self):
        """
        AKDNのCollaborative Part (LightGCN) のための正規化隣接行列を作成します。
        Interaction Graph (User-Item Bipartite Graph) のみを使用します。
        """
        # User -> Item の二部グラフのみを作成 (サイズ: n_entities x n_users)
        rows = [] # Entity (Item) ids
        cols = [] # User ids
        vals = []
        for u_id, items in self.train_user_dict.items():
            # u_id はシフトしていない生の 0 ~ n_users-1
            cols.extend([u_id] * len(items))
            rows.extend(items)
            vals.extend([1.] * len(items))

        adj_u2i = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_items, self.n_users))
        adj_i2u = adj_u2i.T

        # それぞれの行（出次数）で正規化
        user_deg = np.array(adj_u2i.sum(axis=0)).flatten()
        item_deg = np.array(adj_u2i.sum(axis=1)).flatten()
        
        d_u_inv = np.power(user_deg, -0.5)
        d_u_inv[np.isinf(d_u_inv)] = 0.
        d_i_inv = np.power(item_deg, -0.5)
        d_i_inv[np.isinf(d_i_inv)] = 0.
        
        D_u = sp.diags(d_u_inv)
        D_i = sp.diags(d_i_inv)
        
        norm_adj_u2i = D_i.dot(adj_u2i).dot(D_u)
        norm_adj_i2u = D_u.dot(adj_i2u).dot(D_i)
        
        self.norm_adj_user_to_item = self.convert_coo2tensor(norm_adj_u2i.tocoo())
        self.norm_adj_item_to_user = self.convert_coo2tensor(norm_adj_i2u.tocoo())

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        """
        KGEトレーニング用バッチをサンプリングする。

        Args:
            kg_dict:         {h: [(t, r), ...]} の形式の辞書 (self.train_kg_dict)
            batch_size:      バッチサイズ
            highest_neg_idx: negative tail のサンプリング上限 (= n_entities - 1)
        Returns:
            batch_h:     [B] head entity indices
            batch_r:     [B] relation indices
            batch_pos_t: [B] positive tail entity indices
            batch_neg_t: [B] negative tail entity indices (corrupt tail)
        """
        exist_heads = list(kg_dict.keys())
        # head をランダムにバッチサイズ分サンプリング
        if batch_size <= len(exist_heads):
            heads = random.sample(exist_heads, batch_size)
        else:
            heads = [random.choice(exist_heads) for _ in range(batch_size)]

        batch_h, batch_r, batch_pos_t, batch_neg_t = [], [], [], []
        for h in heads:
            # head に対応する (pos_t, r) のペアをランダムに1つ選択
            pos_t, r = random.choice(kg_dict[h])
            # negative tail: エンティティ全体からランダムサンプリング (corrupt tail)
            neg_t = random.randint(0, highest_neg_idx)

            batch_h.append(h)
            batch_r.append(r)
            batch_pos_t.append(pos_t)
            batch_neg_t.append(neg_t)

        batch_h     = torch.LongTensor(batch_h)
        batch_r     = torch.LongTensor(batch_r)
        batch_pos_t = torch.LongTensor(batch_pos_t)
        batch_neg_t = torch.LongTensor(batch_neg_t)
        return batch_h, batch_r, batch_pos_t, batch_neg_t

    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)
