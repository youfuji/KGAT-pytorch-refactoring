import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency in this repo
    jax = None
    jnp = None

if jax is not None:
    from main_t_akdn_jax import load_jax_checkpoint, save_jax_checkpoint
    from model.t_akdn_jax import TAKDNJAX
else:  # pragma: no cover - only used when JAX is unavailable
    load_jax_checkpoint = None
    save_jax_checkpoint = None
    TAKDNJAX = None


@unittest.skipIf(jax is None, "jax is not installed")
class TAKDNJAXSmokeTest(unittest.TestCase):
    def _make_args(self, use_gru_lambda):
        return Namespace(
            embed_dim=8,
            relation_dim=8,
            transr_dim=8,
            conv_dim_list="[8, 8]",
            mess_dropout="[0.0, 0.0]",
            edge_dropout_rate=0.0,
            cf_l2loss_lambda=1e-5,
            tau=1.0,
            lambda_init=0.2,
            lambda_min=0.0,
            lambda_max=1.0,
            lambda_hidden_dim=8,
            use_gru_lambda=use_gru_lambda,
            use_dist_penalty=1,
            use_neighbor_zscore=1,
            use_concat_dist=1,
            jax_disable_jit=1,
            precision="float32",
            lr=1e-3,
        )

    def _make_graph(self):
        return {
            "h_list": np.asarray([0, 0, 1, 2], dtype=np.int32),
            "t_list": np.asarray([1, 2, 2, 0], dtype=np.int32),
            "r_list": np.asarray([0, 1, 0, 1], dtype=np.int32),
            "ig_rows": np.asarray([0, 3, 1, 4, 2, 3], dtype=np.int32),
            "ig_cols": np.asarray([3, 0, 4, 1, 3, 2], dtype=np.int32),
            "ig_vals": np.ones((6,), dtype=np.float32),
            "ig_shape": np.asarray([5, 5], dtype=np.int32),
        }

    def _make_batch(self):
        return {
            "user_ids": jnp.asarray([3, 4], dtype=jnp.int32),
            "item_pos_ids": jnp.asarray([0, 1], dtype=jnp.int32),
            "item_neg_ids": jnp.asarray([2, 2], dtype=jnp.int32),
        }

    def test_forward_and_checkpoint_without_gru(self):
        model = TAKDNJAX(self._make_args(use_gru_lambda=0), 2, 3, 3, 2, self._make_graph())
        params = model.init_params(seed=0)
        batch = self._make_batch()
        loss, aux = model.calc_loss(params, batch, jax.random.PRNGKey(0), 0.2, training=False)

        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(tuple(np.asarray(aux["lambda_records"]).shape), (2,))

        embeddings = model.compute_embeddings(params, jax.random.PRNGKey(1), 0.2, training=False, return_aux=False)
        self.assertEqual(tuple(np.asarray(embeddings).shape), (5, 8))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_jax_checkpoint(params, tmpdir, current_epoch=3)
            restored, epoch = load_jax_checkpoint(str(Path(tmpdir) / "model_epoch3.npz"))
            self.assertEqual(epoch, 3)
            np.testing.assert_allclose(np.asarray(restored["entity_user_embed"]), np.asarray(params["entity_user_embed"]))

    def test_train_step_with_gru(self):
        model = TAKDNJAX(self._make_args(use_gru_lambda=1), 2, 3, 3, 2, self._make_graph())
        params = model.init_params(seed=1)
        opt_state = model.init_optimizer(params)
        batch = self._make_batch()

        new_params, _, aux = model.train_step(params, opt_state, batch, jax.random.PRNGKey(2), 0.2)
        aux = jax.tree_util.tree_map(jax.device_get, aux)

        self.assertTrue(np.isfinite(float(aux["loss"])))
        self.assertEqual(tuple(np.asarray(aux["lambda_records"]).shape), (2,))
        self.assertEqual(np.asarray(new_params["entity_user_embed"]).shape, (5, 8))


if __name__ == "__main__":
    unittest.main()
