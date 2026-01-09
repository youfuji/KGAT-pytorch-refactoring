import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_loader.loader_base import DataLoaderBase


class DataLoaderAKDN(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size
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

        # ユーザーIDのリマッピング（Entity IDとの衝突回避）
        # AKDNでもEmbeddingテーブルを共有する場合に備え、ID空間を分けて管理します
        # 0 ~ n_entities-1 : Entity (Item含む)
        # n_entities ~     : User
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        # CFデータのIDシフト (User ID += n_entities)
        self.cf_train_data = (
            np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32),
            self.cf_train_data[1].astype(np.int32)
        )
        self.cf_test_data = (
            np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32),
            self.cf_test_data[1].astype(np.int32)
        )

        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

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
        # User-Item Interaction Matrixの作成
        # row: User (shifted ID), col: Item (original ID)
        # Note: self.train_user_dict keys are already shifted by n_entities
        
        # 疎行列の構築に必要なリスト
        rows = []
        cols = []
        
        for u_id, items in self.train_user_dict.items():
            rows.extend([u_id] * len(items))
            cols.extend(items)
        
        # Adjacency Matrix A の構築
        # サイズ: (n_users + n_entities) x (n_users + n_entities)
        # 構造: | 0   R |
        #       | R^T 0 |
        # User領域とItem領域(Entity領域の一部)の相互作用
        
        vals = [1.] * len(rows)
        
        # R (User -> Item)
        adj_mat = sp.coo_matrix((vals, (rows, cols)), 
                                shape=(self.n_users_entities, self.n_users_entities))
        
        # R^T (Item -> User)
        adj_mat = adj_mat + adj_mat.T
        
        # 正規化 (D^-1/2 * A * D^-1/2)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        
        # Sparse Tensorに変換して保存
        self.norm_adj_mat = self.convert_coo2tensor(norm_adj_mat.tocoo())

    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)
