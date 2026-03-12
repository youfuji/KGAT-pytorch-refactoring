import torch
import torch.nn as nn
import torch.nn.functional as F

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class T_AKDN(nn.Module):
    """
    T-AKDN: TransR-Enhanced Attention-based Knowledge-aware Deep Network.

    Hybrid attention logit:
      1. L2-normalize e_i, e_v, e_r before TransR projection
      2. e_{i,r} = M_r e_i,  e_{v,r} = M_r e_v
      3. s_sem  = LeakyReLU( e_r^T W_k [e_{v,r} || e_{i,r}] )
      4. s_dist = (1/k) ||e_{i,r} + e_r - e_{v,r}||^2
      5. pi = s_sem - λ * s_dist   (λ is annealed, not learned)
    """

    def __init__(self, args, n_users, n_entities, n_relations, n_items=None, A_in=None,
                 user_pre_embed=None, item_pre_embed=None, edge_dropout_rate=0.0):   
        super(T_AKDN, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items if n_items is not None else n_entities  # アイテムID空間 (0 ~ n_items-1)
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim          # d: entity/user embedding dim
        self.relation_dim = args.relation_dim    # original relation dim (R^d, kept for compatibility)
        self.transr_dim = args.transr_dim        # k: TransR projection dim
        
        self.mess_dropout = eval(args.mess_dropout)
        self.edge_dropout_rate = edge_dropout_rate
        self.n_layers = len(eval(args.conv_dim_list))

        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        self.tau = args.tau  # Attention softmax temperature
        
        # --- T-AKDN specific: λ (annealing, not learned) ---
        self.register_buffer('lambda_val', torch.tensor([0.0], dtype=torch.float))

        # Entity + User Embedding (R^d)
        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
        
        # Relation Embedding (R^d) — 既存互換。他の用途がある場合に備えて残す
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        
        # 初期化 (Xavier)
        nn.init.xavier_uniform_(self.entity_user_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

        # 事前学習済み埋め込みのロード
        if (user_pre_embed is not None) and (item_pre_embed is not None):
            n_pre_items = item_pre_embed.shape[0]
            self.entity_user_embed.weight.data[:n_pre_items].copy_(item_pre_embed)
            self.entity_user_embed.weight.data[self.n_entities : self.n_entities + self.n_users].copy_(user_pre_embed)
        
        # === TransR-Enhanced Attention Parameters ===

        # TransR Projection Matrix M_r: [n_relations, k * d]
        # forward時に view(-1, k, d) して bmm で投影
        self.transr_proj = nn.Embedding(self.n_relations, self.transr_dim * self.embed_dim)
        nn.init.xavier_uniform_(self.transr_proj.weight)

        # Relation Embedding for attention (R^k) — 提案手法のπ計算専用
        self.relation_embed_k = nn.Embedding(self.n_relations, self.transr_dim)
        nn.init.xavier_uniform_(self.relation_embed_k.weight)

        # W_k: Linear(2k -> k) — TransR投影後の連結を入力とする
        self.W_k = nn.Linear(self.transr_dim * 2, self.transr_dim)
        nn.init.xavier_uniform_(self.W_k.weight)

        # === Fusion Gate Parameters (Eq. 4) ===
        self.W_a = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_b = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.W_b.weight)
        
        # IG用隣接行列 (LightGCN用, User-Item Bipartite)
        if A_in is not None:
            self.A_in = nn.Parameter(A_in)
            self.A_in.requires_grad = False
        
        # KG用隣接行列 (Attention付き) は _compute_kg_attention で動的に作成
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
        self.gate_control = 'normal'  # 'normal', 'kg_only', 'ig_only'

    def set_lambda(self, value):
        """λ の値を外部から設定（アニーリング用）"""
        self.lambda_val.fill_(value)

    def set_kg_structure(self, h_list, t_list, r_list, relations):
        """
        KGの構造情報（インデックス）を保存
        """
        self.h_list = h_list
        self.t_list = t_list
        self.r_list = r_list
        self.relations_set = relations
        
        self.n_edges = len(h_list)
        
        # Sparse Matrix用インデックス (2, n_edges)
        self.kg_indices = torch.stack([h_list, t_list], dim=0)

    def _edge_softmax(self, logits, tau=1.0):
        """
        Edge-level softmax per center node with proper autograd support.
        
        Args:
            logits: [E] attention logits per edge
            tau: temperature parameter for sharpness control
        Returns:
            alpha: [E] normalized attention weights (sum-to-1 per center node)
        """
        # Numerical stability: per-head max (if scatter_reduce is available) or clamp
        try:
            head_max = torch.zeros(self.n_entities, device=logits.device, dtype=logits.dtype).fill_(-1e9)
            head_max.scatter_reduce_(0, self.h_list, logits.detach(), reduce='amax')
            logits_stable = (logits - head_max[self.h_list]) / tau
        except AttributeError:
            # Fallback for older PyTorch versions
            logits_stable = torch.clamp(logits / tau, min=-15.0, max=15.0)
        
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
        Hybrid KG Attention (A_kg) を計算 (Differentiable)
        
        提案式:
          1. L2-normalize e_i, e_v, e_r
          2. e_{i,r} = M_r e_i,  e_{v,r} = M_r e_v
          3. s_sem  = LeakyReLU( e_r^T W_k [e_{v,r} || e_{i,r}] )
          4. s_dist = (1/k) ||e_{i,r} + e_r - e_{v,r}||^2
          5. pi = s_sem - λ * s_dist
        
        Args:
            e_entities_curr: 現在の層のEntity Embedding (n_entities, d)
        """
        k = self.transr_dim
        d = self.embed_dim

        # 1. Embedding lookup + L2正規化 (Unit Sphere Constraint)
        # 重要: h = 中心ノード(self/head = e_i), t = 近傍(neighbor/tail = e_v)
        #   既存AKDNコメント: "Tailが近傍(neighbors)、Headが中心"
        h_embed = F.normalize(e_entities_curr[self.h_list], p=2, dim=-1, eps=1e-5)  # [E, d] center (e_i)
        t_embed = F.normalize(e_entities_curr[self.t_list], p=2, dim=-1, eps=1e-5)  # [E, d] neighbor (e_v)
        
        # 2. TransR投影: e_{i,r} = M_r * e_i, e_{v,r} = M_r * e_v
        r_id = self.r_list                                          # [E]
        M = self.transr_proj(r_id).view(-1, k, d)                   # [E, k, d]
        e_ir = torch.bmm(M, h_embed.unsqueeze(-1)).squeeze(-1)      # [E, k] center
        e_vr = torch.bmm(M, t_embed.unsqueeze(-1)).squeeze(-1)      # [E, k] neighbor
        e_r  = F.normalize(self.relation_embed_k(r_id), p=2, dim=-1, eps=1e-5)  # [E, k]
        
        # 3. Semantic score: LeakyReLU( e_r^T * W_k(cat(e_{v,r}, e_{i,r})) )
        # concat順は既存AKDNに合わせて [neighbor, center] = [e_vr, e_ir]
        cat_embed = torch.cat([e_vr, e_ir], dim=-1)                 # [E, 2k]
        q = self.W_k(cat_embed)                                     # [E, k]
        sem = torch.sum(q * e_r, dim=-1)                            # [E]
        sem = self.leakyrelu(sem)                                   # [E]
        
        # 4. Normalized distance: (1/k) * ||e_{i,r} + e_r - e_{v,r}||^2
        dist = torch.sum((e_ir + e_r - e_vr) ** 2, dim=-1) / k     # [E]
        
        # 5. Combined logit: pi = sem - λ * dist  (λ is annealed)
        attention_values = sem - self.lambda_val * dist              # [E]
        
        # 7. Edge-level Softmax with temperature τ
        alpha = self._edge_softmax(attention_values, tau=self.tau)  # [E], sum-to-1 per center node
        
        return alpha  # [E] — sparse tensor を作らず直接返す（勾配保持のため）

    def fusion_gate(self, kg_embed, ig_embed):
        """
        Fusion Gate Mechanism (Eq. 4, 5) - Items Only
        """
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
        \\hat{e}_i^{(l)} = sum( alpha * e_v^{(l-1)} )
        
        sparse.mm ではなく edge-level の scatter 演算を使用。
        これにより alpha を通じて attention パラメータに勾配が流れる。
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
        
        # スケールリセット: 巨大勾配の前層逆流を防ぐ防波堤
        # e_items_kg = F.normalize(e_items_kg, p=2, dim=-1, eps=1e-5)
        
        return e_items_kg

    def _ig_aggregation(self, e_items_dual, e_users_curr):
        """
        IG Aggregation (Eq. 3 & Eq. 6)
        """
        ig_input_ordered = torch.cat([e_items_dual, e_users_curr], dim=0)
        
        if self.training and self.edge_dropout_rate > 0.0:
            A_in = self._sparse_dropout(self.A_in, self.edge_dropout_rate, self.A_in._nnz())
        else:
            A_in = self.A_in

        ig_output = torch.sparse.mm(A_in, ig_input_ordered)
        
        e_items_collab = ig_output[:self.n_entities]
        e_users_new = ig_output[self.n_entities:]
        
        return e_items_collab, e_users_new

    def get_embeddings(self):
        """
        T-AKDNのメインループ (L層の伝播と融合)
        """
        all_embed = self.entity_user_embed.weight
        
        e_entities = all_embed[:self.n_entities]
        e_users = all_embed[self.n_entities:]
        
        user_embeds_list = [e_users]
        item_collab_embeds_list = [e_entities] 
        
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
            # KG Attention + Aggregation + Fusion (最終層はスキップ: dead-end 回避)
            if i < self.n_layers - 1:
                # Step 0: TransR-Enhanced KG Attention
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

            item_collab_embeds_list.append(e_items_collab)
            user_embeds_list.append(e_users_new)
            
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
        if mode == 'calc_kge_loss':
            return self.calc_kge_loss(*input)
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

    def calc_kge_loss(self, h, r, pos_t, neg_t):
        """
        ★ KGE Pairwise Ranking Loss (TransR空間)
        
        TransR投影を用いて正例トリプレットの距離が負例より小さくなるよう最適化:
          L_KGE = mean( softplus( d(h,r,t) - d(h,r,t') ) )
        
        Args:
            h:     [B] head entity indices
            r:     [B] relation indices
            pos_t: [B] positive tail entity indices
            neg_t: [B] negative tail entity indices
        Returns:
            kge_loss: scalar
            l2_loss:  scalar (KGE用 L2 正則化、未加重)
        """
        k = self.transr_dim
        d = self.embed_dim
        all_embed = self.entity_user_embed.weight
        
        # L2正規化 (Unit Sphere Constraint)
        h_e  = F.normalize(all_embed[h],     p=2, dim=-1, eps=1e-5)  # [B, d]
        pt_e = F.normalize(all_embed[pos_t], p=2, dim=-1, eps=1e-5)  # [B, d]
        nt_e = F.normalize(all_embed[neg_t], p=2, dim=-1, eps=1e-5)  # [B, d]
        
        # TransR射影
        M   = self.transr_proj(r).view(-1, k, d)                     # [B, k, d]
        e_r = F.normalize(self.relation_embed_k(r), p=2, dim=-1, eps=1e-5)  # [B, k]
        
        h_proj  = torch.bmm(M, h_e.unsqueeze(-1)).squeeze(-1)       # [B, k]
        pt_proj = torch.bmm(M, pt_e.unsqueeze(-1)).squeeze(-1)      # [B, k]
        nt_proj = torch.bmm(M, nt_e.unsqueeze(-1)).squeeze(-1)      # [B, k]
        
        # TransR距離 (次元正規化あり)
        pos_dist = torch.sum((h_proj + e_r - pt_proj) ** 2, dim=-1) / k  # [B]
        neg_dist = torch.sum((h_proj + e_r - nt_proj) ** 2, dim=-1) / k  # [B]
        
        # Pairwise Ranking Loss: 正例距離 < 負例距離 となるよう最適化
        kge_loss = torch.mean(F.softplus(pos_dist - neg_dist))
        
        # L2 Regularization (KGEタスク用、重み係数は外部で適用)
        l2_loss = _L2_loss_mean(h_e) + _L2_loss_mean(pt_e) + _L2_loss_mean(nt_e)
        
        return kge_loss, l2_loss

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

        # --- ③ 平均有効近傍数（アイテムノードのみ対象） ---
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

        # --- ④ Top-K Attention Ratio（アイテムノードのみからサンプリング） ---
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

    @torch.no_grad()
    def export_item_attention_csv(self, output_path, max_items=None):
        """
        アイテムノードを中心とするエッジのアテンションスコアをCSVに出力。

        出力フォーマット（1行1エッジ）:
          item_id, neighbor_id, relation_id, attention_score

        Args:
            output_path: 出力CSVのパス
            max_items:   出力するアイテムIDの上限 (None=n_items)
        """
        import csv

        e_entities = self.entity_user_embed.weight[:self.n_entities]
        alpha = self._compute_kg_attention(e_entities)  # [E]

        h_cpu = self.h_list.cpu()
        t_cpu = self.t_list.cpu()
        r_cpu = self.r_list.cpu()
        a_cpu = alpha.cpu()

        # アイテムノード (0 ~ n_items-1) のエッジのみ抽出
        limit = max_items if max_items is not None else self.n_items
        item_mask = h_cpu < limit

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['item_id', 'neighbor_id', 'relation_id', 'attention_score'])
            for h, t, r, a in zip(h_cpu[item_mask].tolist(),
                                   t_cpu[item_mask].tolist(),
                                   r_cpu[item_mask].tolist(),
                                   a_cpu[item_mask].tolist()):
                writer.writerow([h, t, r, f'{a:.6f}'])

        n_rows = int(item_mask.sum().item())
        return n_rows
