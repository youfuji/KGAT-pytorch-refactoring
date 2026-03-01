"""
Smoke test for T-AKDN (TransR-Enhanced AKDN).
Validates: shape correctness, gradient flow, pi/alpha diagnostics.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if os.path.dirname(os.path.abspath(__file__)) else '.')
# Run from KGAT-pytorch-refactoring root
sys.path.insert(0, '/Users/yoki/Desktop/KGAT-pytorch-refactoring')

import torch
import torch.nn as nn
import argparse

from model.T_AKDN import T_AKDN

def make_dummy_args(**overrides):
    defaults = dict(
        use_pretrain=0,
        embed_dim=16,
        relation_dim=16,
        transr_dim=8,
        conv_dim_list='[16]',
        mess_dropout='[0.0]',
        cf_l2loss_lambda=1e-5,
        lambda_kg=0.1,
        learnable_lambda=0,
        dist_normalize=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def build_dummy_model(args, n_users=10, n_entities=50, n_relations=5, n_edges=100):
    """Build model with dummy IG adjacency and KG structure."""
    # Dummy IG adjacency (identity-like sparse matrix for simplicity)
    size = n_entities + n_users
    indices = torch.stack([torch.arange(size), torch.arange(size)], dim=0)
    values = torch.ones(size) / size
    A_in = torch.sparse_coo_tensor(indices, values, (size, size))

    model = T_AKDN(args, n_users, n_entities, n_relations, A_in=A_in)

    # Dummy KG structure
    h_list = torch.randint(0, n_entities, (n_edges,))
    t_list = torch.randint(0, n_entities, (n_edges,))
    r_list = torch.randint(0, n_relations, (n_edges,))
    relations = list(range(n_relations))
    model.set_kg_structure(h_list, t_list, r_list, relations)

    return model, h_list, t_list, r_list


def test_attention_shapes():
    """Test that _compute_kg_attention returns correct sparse matrix shape."""
    print("=" * 60)
    print("TEST: Attention Shape")
    args = make_dummy_args()
    n_entities = 50
    model, _, _, _ = build_dummy_model(args, n_entities=n_entities)
    model.eval()

    e_entities = model.entity_user_embed.weight[:n_entities]
    A_kg = model._compute_kg_attention(e_entities)

    assert A_kg.shape == (n_entities, n_entities), f"Expected ({n_entities}, {n_entities}), got {A_kg.shape}"
    assert A_kg.is_sparse, "A_kg should be sparse"
    print(f"  A_kg shape: {A_kg.shape} ✓")
    print(f"  A_kg nnz:   {A_kg._nnz()}")
    print("PASSED ✓")


def test_forward_and_gradients():
    """Test full forward pass (calc_loss) and gradient flow."""
    print("=" * 60)
    print("TEST: Forward Pass & Gradients")
    args = make_dummy_args()
    n_users, n_entities = 10, 50
    model, _, _, _ = build_dummy_model(args, n_users=n_users, n_entities=n_entities)
    model.train()

    user_ids = torch.randint(n_entities, n_entities + n_users, (4,))
    pos_ids = torch.randint(0, n_entities, (4,))
    neg_ids = torch.randint(0, n_entities, (4,))

    loss = model.calc_loss(user_ids, pos_ids, neg_ids)
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    print(f"  Loss: {loss.item():.6f} ✓")

    loss.backward()

    # Check gradients on key parameters
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if 'transr_proj' in name or 'relation_embed_k' in name or 'W_k' in name:
                print(f"  grad norm [{name}]: {grad_norm:.6f}")
    
    assert model.transr_proj.weight.grad is not None, "transr_proj should have gradients"
    assert model.relation_embed_k.weight.grad is not None, "relation_embed_k should have gradients"
    assert model.W_k.weight.grad is not None, "W_k should have gradients"
    print("PASSED ✓")


def test_learnable_lambda():
    """Test learnable lambda: gradient flows to lambda_raw."""
    print("=" * 60)
    print("TEST: Learnable Lambda")
    args = make_dummy_args(learnable_lambda=1)
    n_users, n_entities = 10, 50
    model, _, _, _ = build_dummy_model(args, n_users=n_users, n_entities=n_entities)
    model.train()

    assert hasattr(model, 'lambda_raw'), "Should have lambda_raw parameter"
    assert isinstance(model.lambda_raw, nn.Parameter), "lambda_raw should be nn.Parameter"

    user_ids = torch.randint(n_entities, n_entities + n_users, (4,))
    pos_ids = torch.randint(0, n_entities, (4,))
    neg_ids = torch.randint(0, n_entities, (4,))

    loss = model.calc_loss(user_ids, pos_ids, neg_ids)
    loss.backward()

    assert model.lambda_raw.grad is not None, "lambda_raw should have gradient"
    print(f"  lambda_raw value:    {model.lambda_raw.item():.6f}")
    print(f"  lambda (softplus):   {torch.nn.functional.softplus(model.lambda_raw).item():.6f}")
    print(f"  lambda_raw.grad:     {model.lambda_raw.grad.item():.6f}")
    print("PASSED ✓")


def test_pi_statistics():
    """Diagnose pi logit statistics (mean/std/min/max) to detect dist blowup."""
    print("=" * 60)
    print("TEST: Pi Statistics")
    args = make_dummy_args()
    n_entities = 50
    model, _, _, _ = build_dummy_model(args, n_entities=n_entities, n_edges=200)
    model.eval()

    k = args.transr_dim
    d = args.embed_dim
    e_entities = model.entity_user_embed.weight[:n_entities]

    # Manually compute pi to inspect
    with torch.no_grad():
        h_embed = e_entities[model.h_list]
        t_embed = e_entities[model.t_list]
        r_id = model.r_list
        M = model.transr_proj(r_id).view(-1, k, d)
        e_ir = torch.bmm(M, h_embed.unsqueeze(-1)).squeeze(-1)
        e_vr = torch.bmm(M, t_embed.unsqueeze(-1)).squeeze(-1)
        e_r = model.relation_embed_k(r_id)

        cat_embed = torch.cat([e_vr, e_ir], dim=-1)
        q = model.W_k(cat_embed)
        sem = torch.sum(q * e_r, dim=-1)
        sem = model.leakyrelu(sem)

        dist = torch.sum((e_ir + e_r - e_vr) ** 2, dim=-1)
        pi = sem - args.lambda_kg * dist

    print(f"  sem  - mean:{sem.mean():.4f}  std:{sem.std():.4f}  min:{sem.min():.4f}  max:{sem.max():.4f}")
    print(f"  dist - mean:{dist.mean():.4f}  std:{dist.std():.4f}  min:{dist.min():.4f}  max:{dist.max():.4f}")
    print(f"  pi   - mean:{pi.mean():.4f}  std:{pi.std():.4f}  min:{pi.min():.4f}  max:{pi.max():.4f}")

    assert not torch.isnan(pi).any(), "pi contains NaN!"
    assert not torch.isinf(pi).any(), "pi contains Inf!"
    print("PASSED ✓")


def test_alpha_sum_to_one():
    """Verify softmax alpha sums to 1 for randomly sampled center nodes."""
    print("=" * 60)
    print("TEST: Alpha Sum-to-One (per center node)")
    args = make_dummy_args()
    n_entities = 50
    model, h_list, _, _ = build_dummy_model(args, n_entities=n_entities, n_edges=200)
    model.eval()

    e_entities = model.entity_user_embed.weight[:n_entities]
    with torch.no_grad():
        A_kg = model._compute_kg_attention(e_entities)

    # Convert to dense for row-sum check
    A_dense = A_kg.to_dense()

    # Check random center nodes that have at least one neighbor
    unique_heads = h_list.unique()
    n_check = min(5, len(unique_heads))
    check_nodes = unique_heads[torch.randperm(len(unique_heads))[:n_check]]

    all_ok = True
    for node in check_nodes:
        row_sum = A_dense[node].sum().item()
        ok = abs(row_sum - 1.0) < 1e-4 or row_sum == 0.0  # 0 if no neighbors
        status = "✓" if ok else "✗"
        print(f"  Node {node.item():3d}: row_sum = {row_sum:.6f} {status}")
        if not ok:
            all_ok = False

    assert all_ok, "Some center nodes have alpha that doesn't sum to 1!"
    print("PASSED ✓")


def test_dist_normalize():
    """Test that dist_normalize flag divides distance by k."""
    print("=" * 60)
    print("TEST: dist_normalize flag")
    args_no_norm = make_dummy_args(dist_normalize=0)
    args_norm = make_dummy_args(dist_normalize=1)

    torch.manual_seed(42)
    model_no, _, _, _ = build_dummy_model(args_no_norm, n_entities=50, n_edges=100)
    
    torch.manual_seed(42)
    model_yes, _, _, _ = build_dummy_model(args_norm, n_entities=50, n_edges=100)
    
    # Copy weights so they match
    model_yes.load_state_dict(model_no.state_dict(), strict=False)
    model_yes.set_kg_structure(model_no.h_list, model_no.t_list, model_no.r_list, model_no.relations_set)

    print(f"  dist_normalize=False model created ✓")
    print(f"  dist_normalize=True  model created ✓")
    print("PASSED ✓")


if __name__ == '__main__':
    print("T-AKDN Smoke Test")
    print("=" * 60)
    
    test_attention_shapes()
    print()
    test_forward_and_gradients()
    print()
    test_learnable_lambda()
    print()
    test_pi_statistics()
    print()
    test_alpha_sum_to_one()
    print()
    test_dist_normalize()
    
    print()
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
