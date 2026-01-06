import torch
import torch.nn as nn
import torch.nn.functional as F

class AKDN(nn.Module):
    def __init__(self, args, n_users, n_entities, n_relations, A_in=None,
                 user_pre_embed=None, item_pre_embed=None):   
        super(AKDN, self).__init__()
        self.use_pretrain = args.use_pretrain

        # --- 基本設定 ---
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        
        # LightGCN (IG側) の設定
        self.n_layers = len(eval(args.conv_dim_list))
        self.mess_dropout = eval(args.mess_dropout)
        
        # --- Embedding Layers ---
        # 0 ~ n_entities-1 : Entities (Items含む)
        # n_entities ~ end : Users
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
        
        # --- AKDN Specific Parameters ---
        
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
        
        # Note: Calibration Layerは論文に存在しないため削除
        
        # --- Graph Structure ---
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
        alpha = softmax( LeakyReLU( sum( (W_k[e_v||e_i]) * r ) ) )
        
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
        # ここで LeakyReLU の前に r との積をとるのが論文の式
        # (W_k[...] * r) -> sum -> scalar
        product = trans_embed * r
        attention_logits = torch.sum(product, dim=1)
        
        # 4. Activation
        scores = self.leakyrelu(attention_logits)
        
        return scores

    def update_attention_batch(self, h_list, t_list, r_idx):
        """
        特定のRelationについてAttentionスコアをバッチ計算
        """
        r_embed = self.relation_embed.weight[r_idx] # (dim, )
        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]
        
        # Relation embeddingをバッチサイズに合わせて拡張
        r_embed_batch = r_embed.unsqueeze(0).expand(h_embed.size(0), -1)
        
        return self.calc_kg_attention(h_embed, t_embed, r_embed_batch)

    def update_attention(self, h_list, t_list, r_list, relations):
        """
        Attention重み付き隣接行列 (A_kg) を更新する関数
        """
        device = self.entity_user_embed.weight.device
        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = (r_list == r_idx).nonzero(as_tuple=True)[0]
            if len(index_list) == 0: continue
            
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]
            
            # AKDN Attention Score計算
            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        if len(rows) == 0:
            return

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        # KGのサイズは (n_entities, n_entities)
        shape = (self.n_entities, self.n_entities)
        
        # Sparse Matrix作成
        A_kg = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(device)
        
        # Softmax Normalization (Eq. 2)
        # 行方向(neighbors of h)で正規化
        A_kg = torch.sparse.softmax(A_kg, dim=1)
        
        self.A_kg = A_kg

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

    def get_embeddings(self):
        """
        AKDNのメインループ (L層の伝播と融合)
        Eq. 1, 3, 4, 5, 6 を忠実に実装
        """
        # 初期Embedding (Layer 0)
        # entity_user_embed: [0...n_entities-1] are entities/items, [n_entities...end] are users
        all_embed = self.entity_user_embed.weight
        
        # 分離
        e_entities = all_embed[:self.n_entities]
        e_users = all_embed[self.n_entities:]
        
        # 最終的な表現を格納するリスト (Eq. 7: sum of all layers)
        # 注意: 論文では「collaborative item representation」を使うとあるので、
        # IG由来の表現を蓄積する。Userも同様。
        user_embeds_list = [e_users]
        item_collab_embeds_list = [e_entities] # Layer 0のitemはIG/KG区別なし
        
        # 現在の「Dual Item Representation」 (初期値はそのまま)
        # これは次の層への入力(Userへの伝播)に使われる
        e_items_dual = e_entities
        
        # 現在のUser表現
        e_users_curr = e_users
        
        # 現在のEntity表現 (KG伝播用)
        # 論文では「recursively obtaining... from KG」とあるのでEntityも更新されると解釈
        e_entities_curr = e_entities

        for i in range(self.n_layers):
            # -----------------------------------------------------
            # 1. KG Aggregation (Eq. 1) -> Knowledge-aware Item Rep
            # \hat{e}_i^{(l)} = sum( alpha * e_v^{(l-1)} )
            # -----------------------------------------------------
            if self.A_kg is not None:
                # Sparse MM: (n_ent, n_ent) x (n_ent, dim) -> (n_ent, dim)
                e_items_kg = torch.sparse.mm(self.A_kg, e_entities_curr)
            else:
                e_items_kg = e_entities_curr # Fallback

            # -----------------------------------------------------
            # 2. IG Aggregation (Eq. 3 & Eq. 6)
            # ここでテクニックを使用: LightGCNの行列 A_in を使って同時に更新
            # A_in = [ 0   R ]
            #        [ R^T 0 ]
            # 入力ベクトルを [e_users_curr, e_items_dual] と構成することで:
            #   Top part (User更新): R * e_items_dual -> Dual Itemから集約 (Eq. 6 準拠)
            #   Bottom part (Item更新): R^T * e_users_curr -> Userから集約 (Eq. 3 準拠)
            # -----------------------------------------------------
            
            # 入力ベクトルの結合: [Users, Dual Items]
            # ※ e_items_dualを使うのが重要 (Userへのフィードバック)
            # 入力ベクトルの結合: [Users, Dual Items]
            # ※ e_items_dualを使うのが重要 (Userへのフィードバック)
            
            # A_inは (n_users + n_entities) x (n_users + n_entities)
            # 以前のLoaderでは Userが先かEntityが先か要確認。
            # LoaderAKDNの `create_ig_adjacency` を見ると:
            # rows (User), cols (Item/Entity) -> shape=(n_users+n_entities)
            # 通常、IDは 0~n_ent-1 (Entity), n_ent~ (User) なので、
            # 行列のインデックス順は [Entity, User] の順になっているはずです。
            # よって、結合順序は [Entities(Dual), Users] が正しい。
            
            ig_input_ordered = torch.cat([e_items_dual, e_users_curr], dim=0)
            
            # 伝播
            ig_output = torch.sparse.mm(self.A_in, ig_input_ordered)
            
            # 出力の分離
            e_items_collab = ig_output[:self.n_entities] # Item (Collaborative) \tilde{e}
            e_users_new = ig_output[self.n_entities:]    # User (Updated)
            
            # -----------------------------------------------------
            # 3. Fusion Gate (Eq. 4, 5) -> Dual Item Rep
            # アイテム部分のみに適用
            # -----------------------------------------------------
            e_items_dual_new = self.fusion_gate(e_items_kg, e_items_collab)
            
            # -----------------------------------------------------
            # ストック & 更新
            # -----------------------------------------------------
            # 予測・Loss計算用には IG由来(Collaborative) のItem表現を使う (論文Source 216)
            item_collab_embeds_list.append(e_items_collab)
            user_embeds_list.append(e_users_new)
            
            # 次の層への入力更新
            e_items_dual = e_items_dual_new
            e_users_curr = e_users_new
            
            # Entity更新 (KG側)
            # AKDNはItem表現に注力しているが、多層にするならEntityも更新が必要
            # ここではシンプルに KG Aggregationの結果を次のEntity表現とする(Item兼Entityとして)
            e_entities_curr = e_items_kg 
            
            # -----------------------------------------------------
            # 4. Message Dropout
            # -----------------------------------------------------
            # KGAT同様、各層の出力に対してDropoutを適用
            if self.mess_dropout[i] > 0.0:
                 # リスト内のTensorであれば個別に、単体であればそのまま適用
                 # ここでは e_items_collab (Item) と e_users_new (User) がメインの学習対象
                 e_items_collab = F.dropout(e_items_collab, p=self.mess_dropout[i], training=self.training)
                 e_users_new = F.dropout(e_users_new, p=self.mess_dropout[i], training=self.training)
                 e_items_dual = F.dropout(e_items_dual, p=self.mess_dropout[i], training=self.training)
                 # e_entities_curr にも適用するかは選択によるが、Item側と合わせるのが自然
                 e_entities_curr = F.dropout(e_entities_curr, p=self.mess_dropout[i], training=self.training)
                 
                 # list内の最後の要素をDropout済みのものに更新 (既にappend済みのため)
                 item_collab_embeds_list[-1] = e_items_collab
                 user_embeds_list[-1] = e_users_new 

        # 最終表現 (Eq. 7)
        # collaborative item representation obtained from IG as final item representation
        item_final = torch.stack(item_collab_embeds_list, dim=1).sum(dim=1)
        user_final = torch.stack(user_embeds_list, dim=1).sum(dim=1)
        
        # User, Item の順で結合して返す (他のメソッドとの整合性のため)
        # ただし、IDのマッピングに注意。
        # 呼び出し元は all_embed[user_id] (user_id >= n_entities) とする想定が多い
        # ここでは [Item_Final, User_Final] の形で結合して返す
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
        
        # IDのマッピングに注意
        # user_ids は Loader で既に +n_entities されている
        # item_ids は 0 ~ n_entities-1
        # get_embeddings の戻り値は [Entities(0~N-1), Users(N~)] の順
        
        # 念のためインデックス調整 (もしLoaderのIDがずれていれば)
        # LoaderAKDNでは: cf_train_dataのUser ID += n_entities 済み。
        # よってそのままアクセス可能。
        
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
        
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # L2 Regularization (Eq. 10)
        l2_loss = (user_embed.norm(2).pow(2) + 
                   pos_embed.norm(2).pow(2) + 
                   neg_embed.norm(2).pow(2)) / 2
                   
        return loss + 1e-5 * l2_loss
