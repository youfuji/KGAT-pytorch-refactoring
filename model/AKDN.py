import torch
import torch.nn as nn
import torch.nn.functional as F

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class AKDN(nn.Module):
    def __init__(self, args, n_users, n_entities, n_relations, A_in=None,
                 user_pre_embed=None, item_pre_embed=None, edge_dropout_rate=0.0):   
        super(AKDN, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        
        self.mess_dropout = eval(args.mess_dropout)
        self.edge_dropout_rate = edge_dropout_rate
        self.n_layers = len(eval(args.conv_dim_list))

        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        
        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        
        # 初期化 (Xavier)
        nn.init.xavier_uniform_(self.entity_user_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

        # 事前学習済み埋め込みのロード
        if (user_pre_embed is not None) and (item_pre_embed is not None):
            # Item Part (0 ~ n_items)
            # 事前学習データ(MF)は通常アイテムのみの埋め込みを持つため、対応するID部分のみ更新
            n_pre_items = item_pre_embed.shape[0]
            self.entity_user_embed.weight.data[:n_pre_items].copy_(item_pre_embed)
            
            # User Part (n_entities ~ )
            # ユーザーIDは n_entities から始まるため、そこから user_pre_embed の分だけ更新
            self.entity_user_embed.weight.data[self.n_entities : self.n_entities + self.n_users].copy_(user_pre_embed)
        
        # 1. KG Attention用パラメータ (Eq. 2)
        # W_k: (d || d) -> d  (連結を入力とする)
        self.W_k = nn.Linear(self.embed_dim * 2, self.relation_dim)
        nn.init.xavier_uniform_(self.W_k.weight)
        
        # 2. Fusion Gate用パラメータ (Eq. 4)
        # Gateはアイテムに対してのみ適用される
        self.W_a = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_b = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.W_b.weight)
        
        # IG用隣接行列 (LightGCN用, User-Item Bipartite)
        if A_in is not None:
            self.A_in = nn.Parameter(A_in)
            self.A_in.requires_grad = False
        
        # KG用隣接行列 (Attention付き) は update_attention で作成・保持される
        self.A_kg = None
        
        # Activation
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def calc_kg_attention(self, h, t, r):
        """
        KG側のAttentionスコアを計算 (Eq. 2 準拠)
        alpha = softmax( LeakyReLU( sum( (W_k[e_v * e_i]) * r ) ) )
        
        h: Head items (Batch, dim)
        t: Tail entities (Batch, dim)
        r: Relations (Batch, dim)
        """
        # 1. Concatenate Head & Tail [e_v || e_i] -> (Batch, 2*dim)
        # 実装上の注意: t(tail/neighbor)が e_v, h(head/self)が e_i に相当
        cat_embed = torch.cat([t, h], dim=1)
        
        # 2. Linear Transform W_k -> (Batch, dim)
        trans_embed = self.W_k(cat_embed)
        
        # 3. Relation-aware Interaction (Element-wise Product & Sum)
        # (W_k[...] * r) -> sum -> scalar
        product = trans_embed * r
        attention_logits = torch.sum(product, dim=1)
        
        # 4. Activation
        scores = self.leakyrelu(attention_logits)
        
        return scores

    def set_kg_structure(self, h_list, t_list, r_list, relations):
        """
        KGの構造情報（インデックス）を保存
        """
        self.h_list = h_list
        self.t_list = t_list
        # r_listはRelation EmbeddingのLookupに使う
        self.r_list = r_list
        self.relations_set = relations
        
        # Sparse Matrixのインデックスは静的なので事前に構築しておく
        # rows: h, cols: t
        # ただし、Attention計算後に値を埋め込むために並びを把握しておく必要がある
        # ここでは単純化のため、全エッジに対して一括でAttentionを計算する方式をとる
        
        # エッジ数
        self.n_edges = len(h_list)
        
        # Sparse Matrix用インデックス (2, n_edges)
        self.kg_indices = torch.stack([h_list, t_list], dim=0)

    def _compute_kg_attention(self, e_entities_curr):
        """
        KG Attention (A_kg) を計算 (Differentiable)
        パラメータ W_k, relation_embed, entity_user_embed の勾配が伝播するように計算を行う
        
        Args:
            e_entities_curr: 現在の層のEntity Embedding (n_entities, dim)
        """
        # 1. Embedding lookup
        h_embed = e_entities_curr[self.h_list]
        t_embed = e_entities_curr[self.t_list]
        r_embed = self.relation_embed(self.r_list)
        
        # 2. Attention Score (Eq. 2)
        # alpha = LeakyReLU( W_k([e_t || e_h]) * r ) -> sum
        # Note: AKDNの実装において、Tailが近傍(neighbors)、Headが中心とする
        
        # Concatenate: (n_edges, 2 * dim)
        cat_embed = torch.cat([t_embed, h_embed], dim=1)
        
        # Linear Transform: (n_edges, dim)
        trans_embed = self.W_k(cat_embed)
        
        # Interaction with Relation: (n_edges, dim) -> (n_edges, )
        attention_logits = torch.sum(trans_embed * r_embed, dim=1)
        
        # Activation
        attention_values = self.leakyrelu(attention_logits)
        
        # 3. Create Sparse Matrix & Softmax
        # Sparse Tensor作成 (Valuesに勾配が乗る)
        A_kg_unorm = torch.sparse_coo_tensor(self.kg_indices, attention_values, 
                                             size=(self.n_entities, self.n_entities), 
                                             device=self.kg_indices.device)
        
        # Softmax Normalization (Row-wise)
        # Note: torch.sparse.softmax は dim=1 (row) に対して正規化を行う
        A_kg = torch.sparse.softmax(A_kg_unorm, dim=1)
        
        return A_kg



    def fusion_gate(self, kg_embed, ig_embed):
        """
        Fusion Gate Mechanism (Eq. 4, 5) - Items Only
        """
        # Gate計算 g = sigmoid(W_a * kg + W_b * ig)
        gate_input = self.W_a(kg_embed) + self.W_b(ig_embed)
        g = self.sigmoid(gate_input)
        
        # 融合 e = g * kg + (1-g) * ig
        fused_embed = g * kg_embed + (1 - g) * ig_embed
        return fused_embed

    def _sparse_dropout(self, x, rate, noise_shape):
        """
        Sparse Tensorに対するDropout
        """
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse_coo_tensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def _kg_aggregation(self, A_kg, e_entities_curr):
        """
        KG Aggregation (Eq. 1)
        \hat{e}_i^{(l)} = sum( alpha * e_v^{(l-1)} )
        """
        # Regularization: Edge Dropout (Apply only during training)
        if self.training and self.edge_dropout_rate > 0.0:
            A_kg_drop = self._sparse_dropout(A_kg, self.edge_dropout_rate, A_kg._nnz())
        else:
            A_kg_drop = A_kg

        # Sparse MM: (n_ent, n_ent) x (n_ent, dim) -> (n_ent, dim)
        e_items_kg = torch.sparse.mm(A_kg_drop, e_entities_curr)
        
        return e_items_kg

    def _ig_aggregation(self, e_items_dual, e_users_curr):
        """
        IG Aggregation (Eq. 3 & Eq. 6)
        User Updating: Eq. 6 (Aggregation from Dual Item)
        Item Updating: Eq. 3 (Aggregation from User)
        """
        # 入力ベクトルの結合: [Entities(Dual), Users]
        # 注意: 行列 A_in のインデックス順序は [Entities, Users]
        ig_input_ordered = torch.cat([e_items_dual, e_users_curr], dim=0)
        
        # Regularization: Edge Dropout (Apply only during training)
        if self.training and self.edge_dropout_rate > 0.0:
            A_in = self._sparse_dropout(self.A_in, self.edge_dropout_rate, self.A_in._nnz())
        else:
            A_in = self.A_in

        # 伝播
        ig_output = torch.sparse.mm(A_in, ig_input_ordered)
        
        # 出力の分離
        e_items_collab = ig_output[:self.n_entities] # Item (Collaborative) \tilde{e}
        e_users_new = ig_output[self.n_entities:]    # User (Updated)
        
        return e_items_collab, e_users_new

    def get_embeddings(self):
        """
        AKDNのメインループ (L層の伝播と融合)
        Eq. 1, 3, 4, 5, 6 を忠実に実装
        Refactored version: Aggregation logic is separated into helper methods.
        """
        # 初期Embedding (Layer 0)
        # Note: self.entity_user_embed は _compute_kg_attention ですでに参照されているが、
        # ここでも伝播の起点として使用する
        all_embed = self.entity_user_embed.weight
        
        # 分離
        e_entities = all_embed[:self.n_entities]
        e_users = all_embed[self.n_entities:]
        
        # 最終的な表現を格納するリスト (Eq. 7: sum of all layers)
        user_embeds_list = [e_users]
        item_collab_embeds_list = [e_entities] 
        
        # 現在の「Dual Item Representation」 & User & Entity
        # e_items_dual:  IG入力用 (Fusion後のItem表現)
        # e_users_curr:  IG入力用 (User表現)
        # e_entities_curr: KG入力用 (Entity表現)
        
        e_items_dual = e_entities
        e_users_curr = e_users
        e_entities_curr = e_entities

        for i in range(self.n_layers):
            # Step 0: KG Attention Matrixの計算 (Dynamic & Adaptive)
            # 現在の層のEmbedding (e_entities_curr) に基づいてAttentionを再計算
            A_kg = self._compute_kg_attention(e_entities_curr)

            # 1. KG Aggregation (Eq. 1)
            e_items_kg = self._kg_aggregation(A_kg, e_entities_curr)

            # 2. IG Aggregation (Eq. 3 & Eq. 6)
            e_items_collab, e_users_new = self._ig_aggregation(e_items_dual, e_users_curr)
            
            # 3. Fusion Gate (Eq. 4, 5)
            e_items_dual_new = self.fusion_gate(e_items_kg, e_items_collab)
            
            # -----------------------------------------------------
            # ストック & 更新
            # -----------------------------------------------------
            # 4. Message Dropout
            # 次の層への入力および最終表現のストックに使用する値に対して一貫してDropoutを適用
            if self.mess_dropout[i] > 0.0:
                 e_items_collab = F.dropout(e_items_collab, p=self.mess_dropout[i], training=self.training)
                 e_users_new = F.dropout(e_users_new, p=self.mess_dropout[i], training=self.training)
                 e_items_dual_new = F.dropout(e_items_dual_new, p=self.mess_dropout[i], training=self.training)
                 # e_items_kg = F.dropout(e_items_kg, p=self.mess_dropout[i], training=self.training) # Used only for fusion

            # -----------------------------------------------------
            # ストック & 更新
            # -----------------------------------------------------
            # 予測・Loss計算用には IG由来(Collaborative) のItem表現を使う (論文Source 216)
            item_collab_embeds_list.append(e_items_collab)
            user_embeds_list.append(e_users_new)
            
            # 次の層への入力更新 (Dropout適用後の値を使用)
            e_items_dual = e_items_dual_new
            e_users_curr = e_users_new
            
            # [重要変更]: KG側入力の更新
            # 以前は e_entities_curr = e_items_kg だったが、これだとIG側の情報がKGに伝わらない。
            # Fusionされた情報 (e_items_dual_new) を次のKG入力とすることで、
            # Userの嗜好情報がKG上のEntityへも伝播するようにする (Information Diffusion)
            # e_entities_curr = e_items_dual_new

            # 論文では、KG側の情報にIGの情報は含まれない。
            e_entities_curr = e_items_kg 
            

        # 最終表現 (Eq. 7)
        item_final = torch.stack(item_collab_embeds_list, dim=1).sum(dim=1)
        user_final = torch.stack(user_embeds_list, dim=1).sum(dim=1)
        
        return torch.cat([item_final, user_final], dim=0)

    def forward(self, mode, *input):
        if mode == 'calc_score':
            return self.calc_score(*input)
        if mode == 'calc_loss':
            return self.calc_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)

    def calc_score(self, user_ids, item_ids):
        all_embed = self.get_embeddings()
        user_embed = all_embed[user_ids] 
        item_embed = all_embed[item_ids]
        
        scores = torch.matmul(user_embed, item_embed.transpose(0, 1))
        return scores

    def calc_loss(self, user_ids, item_pos_ids, item_neg_ids):
        all_embed = self.get_embeddings()
        
        user_embed = all_embed[user_ids]
        pos_embed = all_embed[item_pos_ids]
        neg_embed = all_embed[item_neg_ids]
        
        # BPR Loss (Eq. 9)
        pos_scores = torch.sum(user_embed * pos_embed, dim=1)
        neg_scores = torch.sum(user_embed * neg_embed, dim=1)
        
        cf_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # L2 Regularization (Eq. 10)
        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(pos_embed) + _L2_loss_mean(neg_embed)
        return cf_loss + self.cf_l2loss_lambda * l2_loss
