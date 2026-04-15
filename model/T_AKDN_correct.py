import torch
import torch.nn as nn
import torch.nn.functional as F

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class T_AKDN_correct(nn.Module):
    def __init__(self, args, n_users, n_items, n_entities, n_relations,
                 ig_adj_user_to_item=None, ig_adj_item_to_user=None,
                 user_pre_embed=None, item_pre_embed=None, edge_dropout_rate=0.0):   
        super(T_AKDN_correct, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        self.transr_dim = args.transr_dim

        
        self.mess_dropout = eval(args.mess_dropout)
        self.edge_dropout_rate = edge_dropout_rate
        self.n_layers = len(eval(args.conv_dim_list))

        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        
        self.entity_embed = nn.Embedding(self.n_entities, self.embed_dim)
        self.user_embed = nn.Embedding(self.n_users, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        
        # 初期化 (Xavier)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

        # 事前学習済み埋め込みのロード
        if (user_pre_embed is not None) and (item_pre_embed is not None):
            # 事前学習データ(MF)は通常アイテムのみの埋め込みを持つため、対応するID部分のみ更新
            n_pre_items = item_pre_embed.shape[0]
            self.entity_embed.weight.data[:n_pre_items].copy_(item_pre_embed)
            self.user_embed.weight.data.copy_(user_pre_embed)
        
        # === TransR-CKG Parameters ===
        
        # KG Parameters
        self.transr_proj = nn.Embedding(self.n_relations, self.transr_dim * self.embed_dim)
        nn.init.xavier_uniform_(self.transr_proj.weight)

        self.relation_embed_k = nn.Embedding(self.n_relations, self.transr_dim)
        nn.init.xavier_uniform_(self.relation_embed_k.weight)

        self.W_sem = nn.Linear(self.transr_dim * 2, self.transr_dim)
        nn.init.xavier_uniform_(self.W_sem.weight)

        self.W_dist = nn.Linear(self.transr_dim * 3, 1)
        nn.init.xavier_uniform_(self.W_dist.weight)

        self.edge_gate_layer = nn.Linear(2, 1)
        nn.init.xavier_uniform_(self.edge_gate_layer.weight)
        self.sigmoid = nn.Sigmoid()

        # 2. Fusion Gate用パラメータ (Eq. 4)
        # Gateはアイテムに対してのみ適用される
        self.W_a = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_b = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.W_b.weight)
        
        # IG用隣接行列 (LightGCN用, User-Item Bipartite)
        if ig_adj_user_to_item is not None:
            self.ig_adj_user_to_item = nn.Parameter(ig_adj_user_to_item)
            self.ig_adj_user_to_item.requires_grad = False
        if ig_adj_item_to_user is not None:
            self.ig_adj_item_to_user = nn.Parameter(ig_adj_item_to_user)
            self.ig_adj_item_to_user.requires_grad = False
        
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
        self.record_attention = False
        self.attention_records = []
        
        # Ablation Control
        self.gate_control = 'normal' # 'normal', 'kg_only', 'ig_only'

        # デバッグ: 最初の呼び出し時のみループ回数を出力
        self._loop_debug_printed = False
        self._memory_debug_printed = False
        self.last_g_lambda = None
        self.last_layer_g_lambda = None



    def set_kg_structure(self, h_list, t_list, r_list, relations):
        """
        KGの構造情報（インデックス）を保存し、attention高速化用の事前計算を行う
        """
        self.h_list = h_list
        self.t_list = t_list
        self.r_list = r_list
        self.relations_set = relations
        
        # エッジ数
        self.n_edges = len(h_list)
        
        # Sparse Matrix用インデックス (2, n_edges)
        self.kg_indices = torch.stack([h_list, t_list], dim=0)
        
        # 関係ごとのエッジインデックスを事前計算（attention内のunique()+マスク生成を除去）
        self.per_rel_edges = {}
        for r_val in range(int(r_list.max().item()) + 1):
            mask = (r_list == r_val)
            if mask.any():
                self.per_rel_edges[r_val] = mask.nonzero(as_tuple=True)[0]

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
        TransR-CKG Unified KG Attention を計算。
        
        射影行列 M_r は関係ごとに共有されるため、関係ごとの matmul で
        [E, k, d] テンソルの生成を回避する。事前計算済みの per_rel_edges を
        使用し、runtime の unique() + ブールマスクも除去。
        """
        device = e_entities_curr.device
        W_r = self.transr_proj.weight.view(self.n_relations, self.transr_dim, self.embed_dim)
        
        # 全エッジの h/t を一括正規化
        h_all = F.normalize(e_entities_curr[self.h_list], p=2, dim=-1)  # [E, d]
        t_all = F.normalize(e_entities_curr[self.t_list], p=2, dim=-1)  # [E, d]
        
        # 関係ごとに matmul: M_r [k, d] を1つだけ使用（事前計算インデックス）
        e_ir = torch.zeros(self.n_edges, self.transr_dim, device=device)
        e_vr = torch.zeros(self.n_edges, self.transr_dim, device=device)
        
        for r_val, r_edges in self.per_rel_edges.items():
            M_r = W_r[r_val]                            # [k, d] — 16KB のみ
            e_ir[r_edges] = h_all[r_edges] @ M_r.T      # [E_r, d] @ [d, k] → [E_r, k]
            e_vr[r_edges] = t_all[r_edges] @ M_r.T
        
        e_r = F.normalize(self.relation_embed_k(self.r_list), p=2, dim=-1)  # [E, k]
        
        # Semantic Score (s_sem)
        cat_sem = torch.cat([e_vr, e_ir], dim=-1)                           # [E, 2k]
        s_sem = self.leakyrelu(torch.sum(self.W_sem(cat_sem) * e_r, dim=-1)) # [E]
        
        # Distance Score (s_dist)
        cat_dist = torch.cat([e_ir, e_r, e_vr], dim=-1)                     # [E, 3k]
        s_dist = self.leakyrelu(self.W_dist(cat_dist).squeeze(-1))           # [E]
        
        # ====================================================
        # Edge-level Fusion Gate による最適ブレンド
        # ====================================================
        epsilon = 1e-8
        
        # 1. Zスコア正規化 (スケールを揃える)
        sem_norm = (s_sem - s_sem.mean()) / (s_sem.std(unbiased=False) + epsilon)
        dist_norm = (s_dist - s_dist.mean()) / (s_dist.std(unbiased=False) + epsilon)
        
        # 2. Gateネットワークへの入力作成 [E, 2]
        g_lambda_input = torch.stack([sem_norm, dist_norm], dim=-1)
        
        # 3. 割合(g)の算出: 0.0 ~ 1.0
        g_lambda = torch.sigmoid(self.edge_gate_layer(g_lambda_input)).squeeze(-1)
        
        g_lambda_sum = g_lambda.detach().sum()
        g_lambda_count = g_lambda.new_tensor(float(g_lambda.numel())).detach()
        
        attention_values = g_lambda * sem_norm + (1.0 - g_lambda) * dist_norm
        
        # Edge Softmax
        alpha = self._edge_softmax(attention_values)
        
        # デバッグ出力（初回のみ）
        if not self._loop_debug_printed:
            print(f"[TransR attention] n_relations={self.n_relations}, "
                  f"n_edges={self.n_edges}, n_active_relations={len(self.per_rel_edges)}")
            self._loop_debug_printed = True
        
        # Attention 記録（eval時のみ）
        if self.record_attention:
            item_edge_mask = self.h_list < self.n_items
            self.attention_records.append({
                'layer': len(self.attention_records),
                'h': self.h_list[item_edge_mask].detach().cpu(),
                't': self.t_list[item_edge_mask].detach().cpu(),
                'r': self.r_list[item_edge_mask].detach().cpu(),
                'attention_value': attention_values[item_edge_mask].detach().cpu(),
                'alpha': alpha[item_edge_mask].detach().cpu(),
            })
        
        g_lambda_mean = g_lambda_sum / torch.clamp(g_lambda_count, min=1.0)
        return alpha, g_lambda_mean  # [E], scalar



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
        r"""
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

    def _ig_aggregation_item(self, e_users_curr):
        if self.training and self.edge_dropout_rate > 0.0:
            adj_u2i = self._sparse_dropout(self.ig_adj_user_to_item, self.edge_dropout_rate, self.ig_adj_user_to_item._nnz())
        else:
            adj_u2i = self.ig_adj_user_to_item
        e_items_collab = torch.sparse.mm(adj_u2i, e_users_curr)
        return e_items_collab

    def _ig_aggregation_user(self, e_items_dual):
        if self.training and self.edge_dropout_rate > 0.0:
            adj_i2u = self._sparse_dropout(self.ig_adj_item_to_user, self.edge_dropout_rate, self.ig_adj_item_to_user._nnz())
        else:
            adj_i2u = self.ig_adj_item_to_user
        e_users_new = torch.sparse.mm(adj_i2u, e_items_dual)
        return e_users_new

    def get_embeddings(self):
        """
        AKDNのメインループ (L層の伝播と融合)
        Eq. 1, 3, 4, 5, 6 を忠実に実装
        Refactored version: Aggregation logic is separated into helper methods.
        """
        # 初期Embedding (Layer 0)
        e_entities = self.entity_embed.weight
        e_users = self.user_embed.weight
        
        # 最終的な表現を格納するリスト (Eq. 7: sum of all layers)
        user_embeds_list = [e_users]
        item_dual_embeds_list = [e_entities[:self.n_items]]
        
        # e_users_curr:  IG入力用 (User表現)
        # e_entities_curr: KG入力用 (Entity表現)
        
        e_users_curr = e_users
        e_entities_curr = e_entities
        
        if self.record_gate:
            self.gate_coefficients = []
            self.gate_inputs = []
            self.gate_wa_kg = []
            self.gate_wb_ig = []
            self.gate_ig = []
            self.gate_kg = []
        if self.record_attention:
            self.attention_records = []

        layer_g_lambda_means = []

        _do_mem_log = (not self._memory_debug_printed) and e_entities_curr.is_cuda
        def _mem_log(label):
            if not _do_mem_log:
                return
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"  [MEM] {label}: allocated={alloc:.1f}MB, reserved={reserved:.1f}MB")

        for i in range(self.n_layers):
            if _do_mem_log:
                print(f"\n=== Memory Profile: Layer {i} ===")
                _mem_log("layer_start")

            # KG Attention + Aggregation
            alpha, g_lambda_mean = self._compute_kg_attention(e_entities_curr)
            layer_g_lambda_means.append(g_lambda_mean)
            _mem_log("after _compute_kg_attention")

            # 1. KG Aggregation (Eq. 1)
            e_items_kg = self._kg_aggregation(alpha, e_entities_curr)
            _mem_log("after _kg_aggregation")

            # 2. IG Item Aggregation (Eq. 3)
            e_items_collab = self._ig_aggregation_item(e_users_curr)
            _mem_log("after _ig_aggregation_item")
            
            # 3. Fusion Gate (Eq. 4, 5) - アイテムのみに適用
            e_only_items_kg = e_items_kg[:self.n_items]
            e_only_items_collab = e_items_collab[:self.n_items]
            
            # アイテムはKG表現とIG表現を融合
            e_only_items_dual = self.fusion_gate(e_only_items_kg, e_only_items_collab)
            _mem_log("after fusion_gate")
            
            # 4. IG User Aggregation (Eq. 6) - アイテム表現のみを使用
            e_users_new = self._ig_aggregation_user(e_only_items_dual)
            _mem_log("after _ig_aggregation_user")
            
            # 5. Message Dropout
            if self.mess_dropout[i] > 0.0:
                 e_items_collab = F.dropout(e_items_collab, p=self.mess_dropout[i], training=self.training)
                 e_users_new = F.dropout(e_users_new, p=self.mess_dropout[i], training=self.training)
            _mem_log("after dropout")

            # ストック & 更新
            item_dual_embeds_list.append(e_only_items_collab) # 図の通り IG(LightGCN)からの集約情報のアイテム部分のみを使用
            user_embeds_list.append(e_users_new)
            
            # 次の層への入力更新
            e_users_curr = e_users_new
            
            # KG側入力の更新 (論文準拠: KG側にIGの情報は含まない)
            e_entities_curr = e_items_kg

        if _do_mem_log:
            self._memory_debug_printed = True

        if layer_g_lambda_means:
            self.last_layer_g_lambda = torch.stack(layer_g_lambda_means).detach()
            self.last_g_lambda = self.last_layer_g_lambda.mean()
        else:
            self.last_g_lambda = None
            self.last_layer_g_lambda = None
            

        # 最終表現 (Eq. 7)
        item_final = torch.stack(item_dual_embeds_list, dim=1).sum(dim=1)
        user_final = torch.stack(user_embeds_list, dim=1).sum(dim=1)
        
        return user_final, item_final

    def forward(self, mode, *input):
        if mode == 'calc_score':
            return self.calc_score(*input)
        if mode == 'calc_loss':
            return self.calc_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)

    def calc_score(self, user_ids, item_ids):
        user_all_embed, item_all_embed = self.get_embeddings()
        user_embed = user_all_embed[user_ids] 
        item_embed = item_all_embed[item_ids]
        
        scores = torch.matmul(user_embed, item_embed.transpose(0, 1))
        return scores

    def calc_loss(self, user_ids, item_pos_ids, item_neg_ids):
        user_all_embed, item_all_embed = self.get_embeddings()
        
        user_embed = user_all_embed[user_ids]
        pos_embed = item_all_embed[item_pos_ids]
        neg_embed = item_all_embed[item_neg_ids]
        
        # BPR Loss (Eq. 9)
        pos_scores = torch.sum(user_embed * pos_embed, dim=1)
        neg_scores = torch.sum(user_embed * neg_embed, dim=1)
        
        cf_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # L2 Regularization (Eq. 10)
        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(pos_embed) + _L2_loss_mean(neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss

        if self.last_g_lambda is None:
            g_lambda_out = torch.tensor(float('nan'), device=loss.device)
        else:
            g_lambda_out = self.last_g_lambda.to(loss.device)

        if self.last_layer_g_lambda is None:
            layer_g_lambda_out = torch.full((self.n_layers,), float('nan'), device=loss.device)
        else:
            layer_g_lambda_out = self.last_layer_g_lambda.to(loss.device)

        return loss, g_lambda_out, layer_g_lambda_out
        
