from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax


def _l2_loss_mean(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.sum(jnp.square(x), axis=1) / 2.0)


def _as_dtype(precision: str):
    return jnp.float64 if precision == "float64" else jnp.float32


def build_jax_graph_data(data) -> Dict[str, np.ndarray]:
    a_in = data.norm_adj_mat.coalesce()
    ig_indices = a_in.indices().cpu().numpy()
    ig_values = a_in.values().cpu().numpy()

    return {
        "h_list": data.h_list.cpu().numpy().astype(np.int32),
        "t_list": data.t_list.cpu().numpy().astype(np.int32),
        "r_list": data.r_list.cpu().numpy().astype(np.int32),
        "ig_rows": ig_indices[0].astype(np.int32),
        "ig_cols": ig_indices[1].astype(np.int32),
        "ig_vals": ig_values.astype(np.float32),
        "ig_shape": np.asarray(a_in.shape, dtype=np.int32),
    }


class TAKDNJAX:
    def __init__(
        self,
        args,
        n_users: int,
        n_items: int,
        n_entities: int,
        n_relations: int,
        graph_data: Dict[str, np.ndarray],
        user_pre_embed: Optional[np.ndarray] = None,
        item_pre_embed: Optional[np.ndarray] = None,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_users_entities = n_users + n_entities

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        self.transr_dim = args.transr_dim
        self.mess_dropout = tuple(eval(args.mess_dropout))
        self.n_layers = len(eval(args.conv_dim_list))
        self.edge_dropout_rate = float(args.edge_dropout_rate)
        self.cf_l2loss_lambda = float(args.cf_l2loss_lambda)
        self.tau = float(args.tau)
        self.lambda_init = float(getattr(args, "lambda_init", 0.0))
        self.lambda_min = float(getattr(args, "lambda_min", 0.0))
        self.lambda_max = float(getattr(args, "lambda_max", 1.0))
        self.lambda_hidden_dim = int(getattr(args, "lambda_hidden_dim", self.embed_dim))
        self.use_gru_lambda = bool(getattr(args, "use_gru_lambda", 0))
        self.use_dist_penalty = bool(getattr(args, "use_dist_penalty", 1))
        self.use_neighbor_zscore = bool(getattr(args, "use_neighbor_zscore", 1))
        self.use_concat_dist = bool(getattr(args, "use_concat_dist", 1))
        self.use_jit = not bool(getattr(args, "jax_disable_jit", 0))
        self.precision = getattr(args, "precision", "float32")
        self.dtype = _as_dtype(self.precision)

        self.graph = {
            "h_list": jnp.asarray(graph_data["h_list"], dtype=jnp.int32),
            "t_list": jnp.asarray(graph_data["t_list"], dtype=jnp.int32),
            "r_list": jnp.asarray(graph_data["r_list"], dtype=jnp.int32),
            "ig_rows": jnp.asarray(graph_data["ig_rows"], dtype=jnp.int32),
            "ig_cols": jnp.asarray(graph_data["ig_cols"], dtype=jnp.int32),
            "ig_vals": jnp.asarray(graph_data["ig_vals"], dtype=self.dtype),
        }
        self.n_edges = int(graph_data["h_list"].shape[0])

        self._user_pre_embed = user_pre_embed
        self._item_pre_embed = item_pre_embed
        self.optimizer = optax.adam(args.lr)

        if self.use_jit:
            self.train_step = jax.jit(self._train_step_impl)
            self.compute_embeddings = jax.jit(self._compute_embeddings_impl, static_argnames=("training", "return_aux"))
            self.score_from_embeddings = jax.jit(self._score_from_embeddings_impl)
        else:
            self.train_step = self._train_step_impl
            self.compute_embeddings = self._compute_embeddings_impl
            self.score_from_embeddings = self._score_from_embeddings_impl

    def init_params(self, seed: int) -> Dict[str, Any]:
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 16)

        params = {
            "entity_user_embed": self._glorot_uniform(keys[0], (self.n_entities + self.n_users, self.embed_dim)),
            "relation_embed": self._glorot_uniform(keys[1], (self.n_relations, self.relation_dim)),
            "transr_proj": self._glorot_uniform(keys[2], (self.n_relations, self.transr_dim, self.embed_dim)),
            "relation_embed_k": self._glorot_uniform(keys[3], (self.n_relations, self.transr_dim)),
            "W_k": {
                "weight": self._glorot_uniform(keys[4], (self.transr_dim * 2, self.transr_dim)),
                "bias": jnp.zeros((self.transr_dim,), dtype=self.dtype),
            },
            "W_dist": {
                "weight": self._glorot_uniform(keys[5], (self.transr_dim * 3, 1)),
                "bias": jnp.zeros((1,), dtype=self.dtype),
            },
            "W_a": {
                "weight": self._glorot_uniform(keys[6], (self.embed_dim, self.embed_dim)),
            },
            "W_b": {
                "weight": self._glorot_uniform(keys[7], (self.embed_dim, self.embed_dim)),
            },
        }

        if self._item_pre_embed is not None:
            item_pre = jnp.asarray(self._item_pre_embed, dtype=self.dtype)
            params["entity_user_embed"] = params["entity_user_embed"].at[: item_pre.shape[0]].set(item_pre)
        if self._user_pre_embed is not None:
            user_pre = jnp.asarray(self._user_pre_embed, dtype=self.dtype)
            params["entity_user_embed"] = params["entity_user_embed"].at[
                self.n_entities : self.n_entities + user_pre.shape[0]
            ].set(user_pre)

        if self.use_gru_lambda:
            params["lambda_gru"] = {
                "weight_ih": self._glorot_uniform(keys[8], (self.embed_dim * 2, self.lambda_hidden_dim * 3)),
                "weight_hh": self._orthogonal(keys[9], (self.lambda_hidden_dim, self.lambda_hidden_dim * 3)),
                "bias_ih": jnp.zeros((self.lambda_hidden_dim * 3,), dtype=self.dtype),
                "bias_hh": jnp.zeros((self.lambda_hidden_dim * 3,), dtype=self.dtype),
            }
            lambda_ratio = (self.lambda_init - self.lambda_min) / max(self.lambda_max - self.lambda_min, 1e-8)
            lambda_ratio = min(max(lambda_ratio, 1e-4), 1.0 - 1e-4)
            lambda_logit = float(np.log(lambda_ratio / (1.0 - lambda_ratio)))
            params["lambda_out"] = {
                "weight": jnp.zeros((self.lambda_hidden_dim, 1), dtype=self.dtype),
                "bias": jnp.asarray([lambda_logit], dtype=self.dtype),
            }
            params["lambda_h0"] = jnp.zeros((1, self.lambda_hidden_dim), dtype=self.dtype)

        return params

    def init_optimizer(self, params: Dict[str, Any]):
        return self.optimizer.init(params)

    def calc_loss(
        self,
        params: Dict[str, Any],
        batch: Dict[str, jnp.ndarray],
        rng: jax.Array,
        lambda_value: float,
        training: bool,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        all_embed, aux = self.compute_embeddings(params, rng, lambda_value, training=training, return_aux=True)
        user_embed = all_embed[batch["user_ids"]]
        pos_embed = all_embed[batch["item_pos_ids"]]
        neg_embed = all_embed[batch["item_neg_ids"]]

        pos_scores = jnp.sum(user_embed * pos_embed, axis=1)
        neg_scores = jnp.sum(user_embed * neg_embed, axis=1)
        cf_loss = jnp.mean(jax.nn.softplus(neg_scores - pos_scores))
        l2_loss = _l2_loss_mean(user_embed) + _l2_loss_mean(pos_embed) + _l2_loss_mean(neg_embed)
        total_loss = cf_loss + self.cf_l2loss_lambda * l2_loss

        aux_out = {
            "cf_loss": cf_loss,
            "l2_loss": l2_loss,
            "lambda_records": aux["lambda_records"],
        }
        return total_loss, aux_out

    def _train_step_impl(self, params, opt_state, batch, rng, lambda_value):
        def loss_fn(current_params):
            return self.calc_loss(current_params, batch, rng, lambda_value, training=True)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        aux["loss"] = loss
        return new_params, new_opt_state, aux

    def _compute_embeddings_impl(self, params, rng, lambda_value, training=False, return_aux=False):
        all_embed = params["entity_user_embed"]
        e_entities = all_embed[: self.n_entities]
        e_users = all_embed[self.n_entities :]

        user_embeds_list = [e_users]
        item_dual_embeds_list = [e_entities]

        e_items_dual = e_entities
        e_users_curr = e_users
        e_entities_curr = e_entities
        lambda_hidden = params.get("lambda_h0")
        lambda_records = []

        layer_keys = jax.random.split(rng, max(self.n_layers * 8, 1))
        key_idx = 0

        for layer_idx in range(self.n_layers):
            if self.use_gru_lambda:
                layer_lambda, lambda_hidden = self._compute_layer_lambda(params, e_entities_curr, e_users_curr, lambda_hidden)
            else:
                layer_lambda = jnp.asarray(lambda_value, dtype=self.dtype)
            lambda_records.append(layer_lambda)

            alpha = self._compute_kg_attention(params, e_entities_curr, layer_lambda)

            e_items_kg = self._kg_aggregation(
                alpha,
                e_entities_curr,
                layer_keys[key_idx],
                training,
            )
            key_idx += 1

            e_items_collab, e_users_new = self._ig_aggregation(
                e_items_dual,
                e_users_curr,
                layer_keys[key_idx],
                training,
            )
            key_idx += 1

            e_items_dual_new = self._fusion_gate(params, e_items_kg, e_items_collab)

            if self.mess_dropout[layer_idx] > 0.0:
                e_items_collab = self._apply_dropout(
                    e_items_collab, self.mess_dropout[layer_idx], layer_keys[key_idx], training
                )
                key_idx += 1
                e_users_new = self._apply_dropout(
                    e_users_new, self.mess_dropout[layer_idx], layer_keys[key_idx], training
                )
                key_idx += 1
                e_items_dual_new = self._apply_dropout(
                    e_items_dual_new, self.mess_dropout[layer_idx], layer_keys[key_idx], training
                )
                key_idx += 1

            item_dual_embeds_list.append(e_items_dual_new)
            user_embeds_list.append(e_users_new)

            e_items_dual = e_items_dual_new
            e_users_curr = e_users_new
            e_entities_curr = e_items_kg

        item_final = jnp.stack(item_dual_embeds_list, axis=0).sum(axis=0)
        user_final = jnp.stack(user_embeds_list, axis=0).sum(axis=0)
        output = jnp.concatenate([item_final, user_final], axis=0)

        if return_aux:
            aux = {"lambda_records": jnp.stack(lambda_records) if lambda_records else jnp.zeros((0,), dtype=self.dtype)}
            return output, aux
        return output

    def _score_from_embeddings_impl(self, all_embed, user_ids, item_ids):
        user_embed = all_embed[user_ids]
        item_embed = all_embed[item_ids]
        return jnp.matmul(user_embed, item_embed.T)

    def _compute_layer_lambda(self, params, e_entities_curr, e_users_curr, lambda_hidden):
        gru_input = jnp.concatenate(
            [
                jnp.mean(e_entities_curr, axis=0, keepdims=True),
                jnp.mean(e_users_curr, axis=0, keepdims=True),
            ],
            axis=-1,
        )
        lambda_hidden = self._gru_cell(params["lambda_gru"], gru_input, lambda_hidden)
        lambda_logits = jnp.matmul(lambda_hidden, params["lambda_out"]["weight"]) + params["lambda_out"]["bias"]
        lambda_val = self.lambda_min + (self.lambda_max - self.lambda_min) * jax.nn.sigmoid(lambda_logits)
        return lambda_val.squeeze(), lambda_hidden

    def _compute_kg_attention(self, params, e_entities_curr, layer_lambda):
        sem, dist = self._compute_local_scores(params, e_entities_curr, self.graph["h_list"], self.graph["t_list"], self.graph["r_list"])
        sem_norm = self._score_norm(sem)

        if self.use_dist_penalty:
            dist_norm = self._score_norm(dist)
            attention_values = sem_norm + layer_lambda * dist_norm
        else:
            attention_values = sem_norm

        return self._edge_softmax(attention_values)

    def _compute_local_scores(self, params, e_entities_curr, h_idx, t_idx, r_idx):
        h_embed = self._normalize(e_entities_curr[h_idx])
        t_embed = self._normalize(e_entities_curr[t_idx])
        proj = params["transr_proj"][r_idx]
        e_ir = jnp.einsum("ekd,ed->ek", proj, h_embed)
        e_vr = jnp.einsum("ekd,ed->ek", proj, t_embed)
        e_r = self._normalize(params["relation_embed_k"][r_idx])

        cat_embed = jnp.concatenate([e_vr, e_ir], axis=-1)
        q = jnp.matmul(cat_embed, params["W_k"]["weight"]) + params["W_k"]["bias"]
        sem = self._leaky_relu(jnp.sum(q * e_r, axis=-1))

        if self.use_concat_dist:
            dist_input = jnp.concatenate([e_ir, e_r, e_vr], axis=-1)
            dist = self._leaky_relu(
                jnp.matmul(dist_input, params["W_dist"]["weight"]).squeeze(-1) + params["W_dist"]["bias"][0]
            )
        else:
            dist = -jnp.sum(jnp.square(e_ir + e_r - e_vr), axis=-1) / float(self.transr_dim)
        return sem, dist

    def _score_norm(self, values):
        if self.use_neighbor_zscore:
            return self._neighbor_zscore(values)
        mean = jnp.mean(values)
        std = jnp.maximum(jnp.std(values), jnp.asarray(1e-8, dtype=self.dtype))
        return (values - mean) / std

    def _neighbor_zscore(self, values):
        count = jnp.zeros((self.n_entities,), dtype=self.dtype).at[self.graph["h_list"]].add(jnp.ones_like(values))
        count = jnp.maximum(count, jnp.asarray(1.0, dtype=self.dtype))

        mean = jnp.zeros((self.n_entities,), dtype=self.dtype).at[self.graph["h_list"]].add(values) / count
        diff_sq = jnp.square(values - mean[self.graph["h_list"]])
        var = jnp.zeros((self.n_entities,), dtype=self.dtype).at[self.graph["h_list"]].add(diff_sq) / count
        std = jnp.sqrt(var + jnp.asarray(1e-8, dtype=self.dtype))
        return (values - mean[self.graph["h_list"]]) / std[self.graph["h_list"]]

    def _edge_softmax(self, logits):
        neg_inf = jnp.asarray(-1e9, dtype=self.dtype)
        head_max = jnp.full((self.n_entities,), neg_inf, dtype=self.dtype).at[self.graph["h_list"]].max(logits)
        logits_stable = (logits - head_max[self.graph["h_list"]]) / self.tau
        exp_logits = jnp.exp(logits_stable)
        sum_exp = jnp.zeros((self.n_entities,), dtype=self.dtype).at[self.graph["h_list"]].add(exp_logits)
        return exp_logits / (sum_exp[self.graph["h_list"]] + jnp.asarray(1e-16, dtype=self.dtype))

    def _kg_aggregation(self, alpha, e_entities_curr, rng, training):
        if training and self.edge_dropout_rate > 0.0:
            keep = jax.random.bernoulli(rng, 1.0 - self.edge_dropout_rate, shape=alpha.shape).astype(self.dtype)
            alpha = alpha * keep / (1.0 - self.edge_dropout_rate)
        neighbor_embed = e_entities_curr[self.graph["t_list"]]
        weighted = alpha[:, None] * neighbor_embed
        return jnp.zeros((self.n_entities, e_entities_curr.shape[1]), dtype=self.dtype).at[self.graph["h_list"]].add(weighted)

    def _ig_aggregation(self, e_items_dual, e_users_curr, rng, training):
        ig_input = jnp.concatenate([e_items_dual, e_users_curr], axis=0)
        vals = self.graph["ig_vals"]
        if training and self.edge_dropout_rate > 0.0:
            keep = jax.random.bernoulli(rng, 1.0 - self.edge_dropout_rate, shape=vals.shape).astype(self.dtype)
            vals = vals * keep / (1.0 - self.edge_dropout_rate)
        contrib = vals[:, None] * ig_input[self.graph["ig_cols"]]
        output = jnp.zeros((self.n_users_entities, ig_input.shape[1]), dtype=self.dtype).at[self.graph["ig_rows"]].add(contrib)
        return output[: self.n_entities], output[self.n_entities :]

    def _fusion_gate(self, params, kg_embed, ig_embed):
        term_kg = jnp.matmul(kg_embed, params["W_a"]["weight"])
        term_ig = jnp.matmul(ig_embed, params["W_b"]["weight"])
        gate = jax.nn.sigmoid(term_kg + term_ig)
        return gate * kg_embed + (1.0 - gate) * ig_embed

    def _apply_dropout(self, x, p, rng, training):
        if (not training) or p <= 0.0:
            return x
        keep_prob = 1.0 - p
        keep_mask = jax.random.bernoulli(rng, keep_prob, shape=x.shape).astype(self.dtype)
        return x * keep_mask / keep_prob

    def _gru_cell(self, params, x, h):
        gates = jnp.matmul(x, params["weight_ih"]) + params["bias_ih"] + jnp.matmul(h, params["weight_hh"]) + params["bias_hh"]
        r_gate, z_gate, n_gate = jnp.split(gates, 3, axis=-1)
        r_gate = jax.nn.sigmoid(r_gate)
        z_gate = jax.nn.sigmoid(z_gate)

        hh_n = jnp.matmul(h, params["weight_hh"][:, 2 * self.lambda_hidden_dim :]) + params["bias_hh"][2 * self.lambda_hidden_dim :]
        x_n = jnp.matmul(x, params["weight_ih"][:, 2 * self.lambda_hidden_dim :]) + params["bias_ih"][2 * self.lambda_hidden_dim :]
        n_gate = jnp.tanh(x_n + r_gate * hh_n)
        return (1.0 - z_gate) * n_gate + z_gate * h

    def _normalize(self, x):
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x / jnp.maximum(norm, jnp.asarray(1e-5, dtype=self.dtype))

    def _leaky_relu(self, x):
        return jnp.where(x >= 0, x, 0.01 * x)

    def _glorot_uniform(self, key, shape):
        if len(shape) < 2:
            fan_in = fan_out = shape[0]
        else:
            fan_in, fan_out = shape[0], shape[1]
        limit = math.sqrt(6.0 / float(fan_in + fan_out))
        return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=self.dtype)

    def _orthogonal(self, key, shape):
        flat_shape = (shape[0], shape[1])
        a = jax.random.normal(key, flat_shape, dtype=self.dtype)
        q, r = jnp.linalg.qr(a)
        d = jnp.diag(r)
        q *= jnp.sign(d)
        tiled = jnp.tile(q, (1, max(shape[1] // shape[0], 1)))
        return tiled[:, : shape[1]]
