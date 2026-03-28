import argparse


def parse_t_akdn_args():
    parser = argparse.ArgumentParser(description="Run T-AKDN (TransR-enhanced AKDN).")

    parser.add_argument('--seed', type=int, default=2019,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='yelp2018',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book, alibaba-fashion}')
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    parser.add_argument('--cf_batch_size', type=int, default=4096,
                        help='CF batch size.')
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help='Test batch size (the user number to test every batch).')

    parser.add_argument('--embed_dim', type=int, default=64,
                        help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=64,
                        help='Relation Embedding size (original R^d, kept for compatibility).')

    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 64, 64]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--edge_dropout_rate', type=float, default=0.5,
                        help='Dropout probability w.r.t. edge dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    # --- T-AKDN specific hyperparameters ---
    parser.add_argument('--transr_dim', type=int, default=64,
                        help='TransR projection dimension k.')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Temperature parameter for attention softmax. Used as fixed value when --use_gru_tau=0.')
    parser.add_argument('--tau_min', type=float, default=0.1,
                        help='Lower bound for GRU-generated attention temperature.')
    parser.add_argument('--tau_max', type=float, default=10.0,
                        help='Upper bound for GRU-generated attention temperature.')
    parser.add_argument('--tau_hidden_dim', type=int, default=64,
                        help='Hidden size of the GRU used to generate layer-wise tau.')

    # --- Ablation toggle flags (REQUIRED) ---
    parser.add_argument('--use_gru_tau', type=int, required=True, choices=[0, 1],
                        help='1: GRU-based dynamic tau per layer, 0: fixed tau value.')
    parser.add_argument('--use_dist_penalty', type=int, required=True, choices=[0, 1],
                        help='1: add lambda*normalized(dist) to attention logit, 0: semantic score only.')
    parser.add_argument('--use_neighbor_zscore', type=int, required=True, choices=[0, 1],
                        help='1: neighborhood-wise Z-score normalization, 0: global Z-score normalization.')
    parser.add_argument('--use_concat_dist', type=int, required=True, choices=[0, 1],
                        help='1: use concatenation-based dist score, 0: use negative TransR distance.')
    parser.add_argument('--use_lambda_annealing', type=int, required=True, choices=[0, 1],
                        help='1: 3-phase lambda annealing, 0: fixed lambda (uses --lambda_final value).')
    parser.add_argument('--att_chunk_size', type=int, default=0,
                        help='Chunk size for attention computation to prevent OOM. 0 = no chunking (default). '
                             'Recommended: 262144 for Yelp2018.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--lambda_init', type=float, default=0.0,
                        help='Phase 1 lambda value (warmup, sem only).')
    parser.add_argument('--lambda_final', type=float, default=0.5,
                        help='Phase 2-3 lambda target value (saturation).')
    parser.add_argument('--lambda_warmup_epochs', type=int, default=100,
                        help='Phase 1: epochs with lambda=init (no dist penalty).')
    parser.add_argument('--lambda_anneal_epochs', type=int, default=400,
                        help='Phase 2: epochs to linearly anneal lambda from init to final.')

    # --- Attention Diagnostics ---
    parser.add_argument('--attn_diag_threshold', type=float, default=0.35,
                        help='Threshold for effective neighborhood size (alpha > threshold).')
    parser.add_argument('--attn_diag_top_k', type=int, default=2,
                        help='Top-K neighbors for attention ratio diagnostic.')

    parser.add_argument('--n_epoch', type=int, default=500,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing CF loss.')
    parser.add_argument('--evaluate_every', type=int, default=10,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--Ks', nargs='?', default='[20]',
                        help='Calculate metric@K when evaluating.')

    args = parser.parse_args()

    # T-AKDN用の保存ディレクトリ設定（logN で自動採番）
    import os
    base_dir = 'trained_model/T_AKDN_SWITCH/{}/pretrain{}/'.format(
        args.data_name, args.use_pretrain)
    log_count = 0
    while os.path.exists(os.path.join(base_dir, 'log{:d}'.format(log_count))):
        log_count += 1
    save_dir = os.path.join(base_dir, 'log{:d}/'.format(log_count))
    args.save_dir = save_dir

    return args
