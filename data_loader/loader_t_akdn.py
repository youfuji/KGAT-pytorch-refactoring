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
        self.create_ig_adjacency()
        self.print_info(logging)

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
        user_rows = []
        item_cols = []

        for u_id, items in self.train_user_dict.items():
            local_user_id = u_id - self.n_entities
            user_rows.extend([local_user_id] * len(items))
            item_cols.extend(items)

        user_rows = np.asarray(user_rows, dtype=np.int64)
        item_cols = np.asarray(item_cols, dtype=np.int64)
        vals = np.ones_like(user_rows, dtype=np.float32)

        user_degree = np.bincount(user_rows, minlength=self.n_users).astype(np.float32)
        item_degree = np.bincount(item_cols, minlength=self.n_entities).astype(np.float32)
        norm_values = np.power(user_degree[user_rows] * item_degree[item_cols], -0.5)
        norm_values[np.isinf(norm_values)] = 0.0

        user_to_item = sp.coo_matrix(
            (norm_values, (item_cols, user_rows)),
            shape=(self.n_entities, self.n_users),
        )
        item_to_user = sp.coo_matrix(
            (norm_values, (user_rows, item_cols)),
            shape=(self.n_users, self.n_entities),
        )

        self.norm_adj_user_to_item = self.convert_coo2tensor(user_to_item)
        self.norm_adj_item_to_user = self.convert_coo2tensor(item_to_user)

        # Keep the original full bipartite adjacency for compatibility and debugging.
        full_rows = np.concatenate([item_cols, user_rows + self.n_entities])
        full_cols = np.concatenate([user_rows + self.n_entities, item_cols])
        full_vals = np.concatenate([norm_values, norm_values])
        full_adj = sp.coo_matrix(
            (full_vals, (full_rows, full_cols)),
            shape=(self.n_users_entities, self.n_users_entities),
        )
        self.norm_adj_mat = self.convert_coo2tensor(full_adj)

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
        logging.info('n_ig_user_item:    %d' % self.norm_adj_user_to_item._nnz())
        logging.info('n_ig_item_user:    %d' % self.norm_adj_item_to_user._nnz())
