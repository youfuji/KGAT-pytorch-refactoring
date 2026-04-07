import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_loader.loader_base import DataLoaderBase


class DataLoaderTAKDN(DataLoaderBase):

    IG_RELATION_USER_TO_ITEM = 0
    IG_RELATION_ITEM_TO_USER = 1

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)
        self.create_ig_adjacency()

    def construct_data(self, kg_data):
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # Reserve 2 relation ids for IG edges:
        # 0: user -> item, 1: item -> user
        kg_data['r'] += 2

        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (
            np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32),
            self.cf_train_data[1].astype(np.int32),
        )
        self.cf_test_data = (
            np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32),
            self.cf_test_data[1].astype(np.int32),
        )

        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        # Keep KG structure pure KG-only. IG edges are handled separately.
        self.kg_train_data = kg_data
        self.n_kg_train = len(self.kg_train_data)

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
        rows = []
        cols = []

        for u_id, items in self.train_user_dict.items():
            rows.extend([u_id] * len(items))
            cols.extend(items)

        vals = [1.0] * len(rows)

        adj_mat = sp.coo_matrix(
            (vals, (rows, cols)),
            shape=(self.n_users_entities, self.n_users_entities),
        )
        adj_mat = adj_mat + adj_mat.T

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
        self.norm_adj_mat = self.convert_coo2tensor(norm_adj_mat)

        ig_rows = norm_adj_mat.row.astype(np.int64)
        ig_cols = norm_adj_mat.col.astype(np.int64)
        ig_values = norm_adj_mat.data.astype(np.float32)

        ig_relations = np.full_like(ig_rows, fill_value=self.IG_RELATION_ITEM_TO_USER)
        user_to_item_mask = ig_rows >= self.n_entities
        ig_relations[user_to_item_mask] = self.IG_RELATION_USER_TO_ITEM

        self.ig_edge_index = torch.LongTensor(np.vstack((ig_rows, ig_cols)))
        self.ig_relation_ids = torch.LongTensor(ig_relations)
        self.ig_edge_values = torch.FloatTensor(ig_values)

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
        logging.info('n_ig_edges:        %d' % self.ig_edge_values.numel())
