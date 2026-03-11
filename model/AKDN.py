import torch
import torch.nn as nn
import torch.nn.functional as F

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class AKDN(nn.Module):
    def __init__(self, args, n_users, n_entities, n_relations, n_items=None, A_in=None,
                 user_pre_embed=None, item_pre_embed=None, edge_dropout_rate=0.0):   
        super(AKDN, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items if n_items is not None else n_entities  # アイテムID空間 (0 ~ n_items-1)
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
        
        # 可視化用
        self.record_gate = False
        self.gate_coefficients = []
        self.gate_inputs = []
        self.gate_wa_kg = []
        self.gate_wb_ig = []
        self.gate_ig = []
        self.gate_kg = []
        
        # Ablation Control
        self.gate_control = 'normal' # 'normal', 'kg_only', 'ig_only'

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

    def _edge_softmax(self, logits):
        """
        Edge-level softmax per center node with proper autograd support.
        
        Args:
            logits: [E] attention logits per edge
        Returns:
            alpha: [E] normalized attention weights (sum-to-1 per center node)
        """
        # Numerical stability: per-head max subtraction
        try:
            head_max = torch.zeros(self.n_entities, device=logits.device, dtype=logits.dtype).fill_(-1e9)
            head_max.scatter_reduce_(0, self.h_list, logits.detach(), reduce='amax')
            logits_stable = logits - head_max[self.h_list]
        except AttributeError:
            # Fallback for older PyTorch versions
            logits_stable = torch.clamp(logits, min=-15.0, max=15.0)
        
        # Exponentiate
        exp_logits = torch.exp(logits_stable)
        
        # Per-center-node sum (out-of-place index_add for proper autograd)
        sum_exp = torch.zeros(self.n_entities, device=logits.device, dtype=logits.dtype)
        sum_exp = sum_exp.index_add(0, self.h_list, exp_logits)
        
        # Normalize
        alpha = exp_logits / (sum_exp[self.h_list] + 1e-16)
        return alpha

    def _compute_kg_attention(self, e_entities_curr):
        """
        KG Attention を計算 (Differentiable)
        パラメータ W_k, relation_embed, entity_user_embed の勾配が伝播するように計算を行う
        
        Args:
            e_entities_curr: 現在の層のEntity Embedding (n_entities, dim)
        Returns:
            alpha: [E] edge-level attention weights
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
        
        # 3. Edge Softmax (per-head normalization)
        alpha = self._edge_softmax(attention_values)
        
        return alpha  # [E]



    def fusion_gate(self, kg_embed, ig_embed):
        """
        Fusion Gate Mechanism (Eq. 4, 5) - Items Only
        """
        # Gate計算 g = sigmoid(W_a * kg + W_b * ig)
        # Gate計算 g = sigmoid(W_a * kg + W_b * ig)
        term_kg = self.W_a(kg_embed)
        term_ig = self.W_b(ig_embed)
        gate_input = term_kg + term_ig
        g = self.sigmoid(gate_input)
        
        if self.record_gate:
            self.gate_coefficients.append(g.detach().cpu())
            self.gate_inputs.append(gate_input.detach().cpu())
            self.gate_wa_kg.append(term_kg.detach().cpu())
            self.gate_wb_ig.append(term_ig.detach().cpu())
            self.gate_ig.append(ig_embed.detach().cpu())
            self.gate_kg.append(kg_embed.detach().cpu())
            
        # Ablation Logic
        if self.gate_control == 'kg_only':
            g = torch.ones_like(g)
        elif self.gate_control == 'ig_only':
            g = torch.zeros_like(g)
        
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

    def _kg_aggregation(self, alpha, e_entities_curr):
        """
        KG Aggregation (Eq. 1)
        \hat{e}_i^{(l)} = sum( alpha * e_v^{(l-1)} )
        
        scatter 演算を使用し、alpha を通じて attention パラメータに勾配が流れる。
        """
        # Edge dropout (edge-level)
        if self.training and self.edge_dropout_rate > 0.0:
            drop_mask = (torch.rand(alpha.size(0), device=alpha.device) >= self.edge_dropout_rate).float()
            alpha = alpha * drop_mask / (1.0 - self.edge_dropout_rate)
        
        # Scatter-based aggregation: e_i = sum_j alpha_{ij} * e_j
        neighbor_embed = e_entities_curr[self.t_list]                       # [E, d]
        weighted = alpha.unsqueeze(-1) * neighbor_embed                    # [E, d]
        e_items_kg = torch.zeros(self.n_entities, e_entities_curr.size(1),
                                 device=e_entities_curr.device)
        e_items_kg = e_items_kg.index_add(0, self.h_list, weighted)        # [N, d]
        
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
        
        if self.record_gate:
            self.gate_coefficients = []
            self.gate_inputs = []
            self.gate_wa_kg = []
            self.gate_wb_ig = []
            self.gate_ig = []
            self.gate_kg = []

        for i in range(self.n_layers):
            # KG Attention + Aggregation (最終層はスキップ: dead-end 回避)
            if i < self.n_layers - 1:
                # Step 0: KG Attention の計算 (Dynamic & Adaptive)
                alpha = self._compute_kg_attention(e_entities_curr)

                # 1. KG Aggregation (Eq. 1)
                e_items_kg = self._kg_aggregation(alpha, e_entities_curr)

            # 2. IG Aggregation (Eq. 3 & Eq. 6)
            e_items_collab, e_users_new = self._ig_aggregation(e_items_dual, e_users_curr)
            
            # 3. Fusion Gate (Eq. 4, 5) — 最終層はIG出力をそのまま使用
            if i < self.n_layers - 1:
                e_items_dual_new = self.fusion_gate(e_items_kg, e_items_collab)
            else:
                e_items_dual_new = e_items_collab
            
            # 4. Message Dropout
            if self.mess_dropout[i] > 0.0:
                 e_items_collab = F.dropout(e_items_collab, p=self.mess_dropout[i], training=self.training)
                 e_users_new = F.dropout(e_users_new, p=self.mess_dropout[i], training=self.training)
                 e_items_dual_new = F.dropout(e_items_dual_new, p=self.mess_dropout[i], training=self.training)

            # ストック & 更新
            item_collab_embeds_list.append(e_items_collab)
            user_embeds_list.append(e_users_new)
            
            # 次の層への入力更新
            e_items_dual = e_items_dual_new
            e_users_curr = e_users_new
            
            # KG側入力の更新 (論文準拠: KG側にIGの情報は含まない)
            if i < self.n_layers - 1:
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

    @torch.no_grad()
    def compute_attention_diagnostics(self, threshold=0.35, top_k=2, max_sample_nodes=1000):
        """
        アテンション形状の診断指標を計算（推論モード、勾配不要）。

        第1層の alpha を取得し、中心ノードごとに以下を算出:
          - effective_neighbors: 閾値超えの有効近傍数の平均
          - topk_ratio: 上位K個の重み占有率の平均

        Args:
            threshold: 有効近傍とみなす alpha の閾値 (default: 0.35)
            top_k: Top-K Ratio を計算する上位個数 (default: 2)
            max_sample_nodes: Top-K Ratio 計算時のランダムサンプリング上限 (default: 1000)
        Returns:
            dict: {'effective_neighbors': float, 'topk_ratio': float}
        """
        # 第1層のEntity Embeddingでalphaを計算
        e_entities = self.entity_user_embed.weight[:self.n_entities]
        alpha = self._compute_kg_attention(e_entities)  # [E]

        # --- 平均有効近傍数（アイテムノードのみ対象） ---
        effective_mask = (alpha > threshold).float()  # [E]
        eff_per_node = torch.zeros(self.n_entities, device=alpha.device)
        eff_per_node.index_add_(0, self.h_list, effective_mask)

        # アイテムノードの範囲 (0 ~ n_items-1) に絞って平均をとる
        has_neighbors = torch.zeros(self.n_entities, device=alpha.device)
        has_neighbors.index_add_(0, self.h_list, torch.ones_like(alpha))
        item_eff = eff_per_node[:self.n_items]
        item_has = has_neighbors[:self.n_items]
        active_mask = item_has > 0
        if active_mask.sum() > 0:
            avg_effective_neighbors = item_eff[active_mask].mean().item()
        else:
            avg_effective_neighbors = 0.0

        # --- Top-K Attention Ratio（アイテムノードのみからサンプリング） ---
        unique_heads = self.h_list.unique()
        # アイテムID (0 ~ n_items-1) のみに絞る
        item_heads = unique_heads[unique_heads < self.n_items]
        n_unique = item_heads.size(0)

        # サンプリング: アイテムノード数が多い場合はランダムに絞る
        if n_unique > max_sample_nodes:
            perm = torch.randperm(n_unique, device=item_heads.device)[:max_sample_nodes]
            sampled_heads = item_heads[perm]
        else:
            sampled_heads = item_heads

        topk_ratios = []
        for head in sampled_heads:
            edge_mask = (self.h_list == head)
            head_alpha = alpha[edge_mask]
            k_actual = min(top_k, head_alpha.size(0))
            topk_vals, _ = head_alpha.topk(k_actual)
            topk_ratios.append(topk_vals.sum().item())

        avg_topk_ratio = sum(topk_ratios) / len(topk_ratios) if topk_ratios else 0.0

        return {
            'effective_neighbors': avg_effective_neighbors,
            'topk_ratio': avg_topk_ratio,
        }

