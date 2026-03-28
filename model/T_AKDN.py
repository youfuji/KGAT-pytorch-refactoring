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
            5. pi = Z-score(s_sem) + λ * Z-score(-s_dist)   (λ is annealed, not learned)
    """

    def __init__(self, args, n_users, n_items, n_entities, n_relations, A_in=None,
                 user_pre_embed=None, item_pre_embed=None, edge_dropout_rate=0.0):

        super(T_AKDN, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim          # d: entity/user embedding dim
        self.relation_dim = args.relation_dim    # original relation dim (R^d, kept for compatibility)
        self.transr_dim = args.transr_dim        # k: TransR projection dim
        
        self.mess_dropout = eval(args.mess_dropout)
        self.edge_dropout_rate = edge_dropout_rate
        self.n_layers = len(eval(args.conv_dim_list))

        # Attention chunk size for OOM prevention (0 = no chunking)
        self.att_chunk_size = int(getattr(args, 'att_chunk_size', 0))

        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        self.tau_init = float(args.tau)
        self.tau_min = float(getattr(args, 'tau_min', 0.1))
        self.tau_max = float(getattr(args, 'tau_max', 10.0))
        self.tau_hidden_dim = int(getattr(args, 'tau_hidden_dim', self.embed_dim))

        # --- Ablation toggle flags ---
        self.use_gru_tau = bool(getattr(args, 'use_gru_tau', 1))
        self.use_dist_penalty = bool(getattr(args, 'use_dist_penalty', 1))
        self.use_neighbor_zscore = bool(getattr(args, 'use_neighbor_zscore', 1))
        
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

        # GRU-based temperature controller: layer-wise tau_l を生成する
        if self.use_gru_tau:
            self.tau_gru = nn.GRUCell(self.embed_dim * 2, self.tau_hidden_dim)
            self.tau_out = nn.Linear(self.tau_hidden_dim, 1)
            self.tau_h0 = nn.Parameter(torch.zeros(1, self.tau_hidden_dim))
            nn.init.xavier_uniform_(self.tau_gru.weight_ih)
            nn.init.orthogonal_(self.tau_gru.weight_hh)
            nn.init.zeros_(self.tau_gru.bias_ih)
            nn.init.zeros_(self.tau_gru.bias_hh)
            nn.init.zeros_(self.tau_out.weight)
            tau_ratio = (self.tau_init - self.tau_min) / max(self.tau_max - self.tau_min, 1e-8)
            tau_ratio = min(max(tau_ratio, 1e-4), 1.0 - 1e-4)
            tau_logit = torch.logit(torch.tensor([tau_ratio], dtype=torch.float))
            self.tau_out.bias = nn.Parameter(tau_logit)

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
        self.record_attention = False
        self.attention_records = []
        self.tau_records = []
        
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

    def _neighbor_zscore(self, values):
        """
        Z-score normalize edge values within each center node neighborhood.

        Args:
            values: [E] edge-wise values aligned with self.h_list
        Returns:
            [E] z-scored values per center node
        """
        ones = torch.ones_like(values)
        count = torch.zeros(self.n_entities, device=values.device, dtype=values.dtype)
        count = count.index_add(0, self.h_list, ones).clamp(min=1)  # [N]

        mean = torch.zeros(self.n_entities, device=values.device, dtype=values.dtype)
        mean = mean.index_add(0, self.h_list, values) / count       # [N]

        diff_sq = (values - mean[self.h_list]) ** 2                 # [E]
        var = torch.zeros(self.n_entities, device=values.device, dtype=values.dtype)
        var = var.index_add(0, self.h_list, diff_sq) / count        # [N]
        std = (var + 1e-8).sqrt()                                   # [N]

        return (values - mean[self.h_list]) / std[self.h_list]      # [E]

    def _compute_layer_tau(self, e_entities_curr, e_users_curr, tau_hidden):
        """
        Layer-wise GRU state から attention temperature tau_l を生成する。
        """
        gru_input = torch.cat([
            e_entities_curr.mean(dim=0, keepdim=True),
            e_users_curr.mean(dim=0, keepdim=True),
        ], dim=-1)
        tau_hidden = self.tau_gru(gru_input, tau_hidden)
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_out(tau_hidden))
        return tau.squeeze(), tau_hidden

    def _compute_local_scores(self, e_entities_curr, h_idx, t_idx, r_idx):
        """
        エッジのサブセットに対して sem と dist を計算する（ローカル演算のみ）。
        Z-score / softmax は全エッジ結合後に呼び出し側で行う。

        Returns:
            sem:  [len(h_idx)] semantic scores
            dist: [len(h_idx)] distance scores
        """
        k = self.transr_dim
        d = self.embed_dim

        h_embed = F.normalize(e_entities_curr[h_idx], p=2, dim=-1, eps=1e-5)
        t_embed = F.normalize(e_entities_curr[t_idx], p=2, dim=-1, eps=1e-5)

        M = self.transr_proj(r_idx).view(-1, k, d)
        e_ir = torch.bmm(M, h_embed.unsqueeze(-1)).squeeze(-1)
        e_vr = torch.bmm(M, t_embed.unsqueeze(-1)).squeeze(-1)
        e_r  = F.normalize(self.relation_embed_k(r_idx), p=2, dim=-1, eps=1e-5)

        cat_embed = torch.cat([e_vr, e_ir], dim=-1)
        q = self.W_k(cat_embed)
        sem = torch.sum(q * e_r, dim=-1)
        sem = self.leakyrelu(sem)

        dist = torch.sum((e_ir + e_r - e_vr) ** 2, dim=-1) / k

        return sem, dist

    def _compute_kg_attention(self, e_entities_curr, tau):
        """
        Hybrid KG Attention (A_kg) を計算 (Differentiable)
        
        提案式:
          1. L2-normalize e_i, e_v, e_r
          2. e_{i,r} = M_r e_i,  e_{v,r} = M_r e_v
          3. s_sem  = LeakyReLU( e_r^T W_k [e_{v,r} || e_{i,r}] )
          4. s_dist = (1/k) ||e_{i,r} + e_r - e_{v,r}||^2
                    5. sem_tilde = Z-score(s_sem)      per center node
                    6. d_tilde_neg = Z-score(-s_dist)  per center node
                    7. pi = sem_tilde + λ * d_tilde_neg
        
        Args:
            e_entities_curr: 現在の層のEntity Embedding (n_entities, d)
        """
        E = self.n_edges
        C = self.att_chunk_size

        if C > 0 and E > C:
            # --- チャンク分割モード ---
            sem_chunks = []
            dist_chunks = []
            for start in range(0, E, C):
                end = min(start + C, E)
                h_c = self.h_list[start:end]
                t_c = self.t_list[start:end]
                r_c = self.r_list[start:end]
                sem_c, dist_c = self._compute_local_scores(e_entities_curr, h_c, t_c, r_c)
                sem_chunks.append(sem_c)
                dist_chunks.append(dist_c)
            sem  = torch.cat(sem_chunks, dim=0)   # [E]
            dist = torch.cat(dist_chunks, dim=0)  # [E]
        else:
            # --- 一括処理モード（従来動作） ---
            sem, dist = self._compute_local_scores(
                e_entities_curr, self.h_list, self.t_list, self.r_list)

        # Combine scores based on ablation flags
        if self.use_neighbor_zscore:
            sem_tilde   = self._neighbor_zscore(sem)       # [E]
            if self.use_dist_penalty:
                d_tilde_neg = self._neighbor_zscore(-dist)     # [E]
                attention_values = sem_tilde + self.lambda_val * d_tilde_neg  # [E]
            else:
                attention_values = sem_tilde  # [E]
        else:
            if self.use_dist_penalty:
                attention_values = sem + self.lambda_val * (-dist)  # [E]
            else:
                attention_values = sem  # [E]
        
        # Edge-level Softmax with temperature τ_l
        alpha = self._edge_softmax(attention_values, tau=tau)  # [E]

        if self.record_attention:
            item_edge_mask = self.h_list < self.n_items
            record = {
                'layer': len(self.attention_records),
                'tau': float(tau.detach().cpu()) if torch.is_tensor(tau) else float(tau),
                'h': self.h_list[item_edge_mask].detach().cpu(),
                't': self.t_list[item_edge_mask].detach().cpu(),
                'r': self.r_list[item_edge_mask].detach().cpu(),
                'sem': sem[item_edge_mask].detach().cpu(),
                'dist': dist[item_edge_mask].detach().cpu(),
                'alpha': alpha[item_edge_mask].detach().cpu(),
            }
            if self.use_neighbor_zscore:
                record['sem_tilde'] = sem_tilde[item_edge_mask].detach().cpu()
                if self.use_dist_penalty:
                    record['d_tilde_neg'] = d_tilde_neg[item_edge_mask].detach().cpu()
            self.attention_records.append(record)
        
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
        
        E = alpha.size(0)
        C = self.att_chunk_size
        d = e_entities_curr.size(1)

        e_items_kg = torch.zeros(self.n_entities, d, device=e_entities_curr.device)

        if C > 0 and E > C:
            # --- チャンク分割モード ---
            for start in range(0, E, C):
                end = min(start + C, E)
                t_c = self.t_list[start:end]
                neighbor_embed = e_entities_curr[t_c]                        # [C, d]
                weighted = alpha[start:end].unsqueeze(-1) * neighbor_embed   # [C, d]
                e_items_kg = e_items_kg.index_add(0, self.h_list[start:end], weighted)
        else:
            # --- 一括処理モード（従来動作） ---
            neighbor_embed = e_entities_curr[self.t_list]                    # [E, d]
            weighted = alpha.unsqueeze(-1) * neighbor_embed                 # [E, d]
            e_items_kg = e_items_kg.index_add(0, self.h_list, weighted)      # [N, d]
        
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
        item_dual_embeds_list = [e_entities]
        
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
        self.tau_records = []
        if self.record_attention:
            self.attention_records = []

        tau_hidden = self.tau_h0.expand(1, -1) if self.use_gru_tau else None

        for i in range(self.n_layers):
            # KG Attention + Aggregation + Fusion
            # 最終層でもKG側を計算し、fused item表現に反映する。
            if self.use_gru_tau:
                tau, tau_hidden = self._compute_layer_tau(e_entities_curr, e_users_curr, tau_hidden)
            else:
                tau = self.tau_init
            self.tau_records.append(float(tau.detach().cpu()) if torch.is_tensor(tau) else float(tau))
            alpha = self._compute_kg_attention(e_entities_curr, tau)

            # 1. KG Aggregation (Eq. 1)
            e_items_kg = self._kg_aggregation(alpha, e_entities_curr)

            # 2. IG Aggregation (Eq. 3 & Eq. 6)
            e_items_collab, e_users_new = self._ig_aggregation(e_items_dual, e_users_curr)
            
            # 3. Fusion Gate (Eq. 4, 5)
            e_items_dual_new = self.fusion_gate(e_items_kg, e_items_collab)
            
            # 4. Message Dropout
            if self.mess_dropout[i] > 0.0:
                 e_items_collab = F.dropout(e_items_collab, p=self.mess_dropout[i], training=self.training)
                 e_users_new = F.dropout(e_users_new, p=self.mess_dropout[i], training=self.training)
                 e_items_dual_new = F.dropout(e_items_dual_new, p=self.mess_dropout[i], training=self.training)

            item_dual_embeds_list.append(e_items_dual_new)
            user_embeds_list.append(e_users_new)
            
            e_items_dual = e_items_dual_new
            e_users_curr = e_users_new
            
            # KG側入力の更新 (論文準拠: KG側にIGの情報は含まない)
            e_entities_curr = e_items_kg 
            

        # 最終表現 (Eq. 7)
        item_final = torch.stack(item_dual_embeds_list, dim=1).sum(dim=1)
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
