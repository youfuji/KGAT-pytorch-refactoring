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
                        help='Fixed temperature parameter for attention softmax.')

    # --- Ablation toggle flags (REQUIRED) ---
        # 1ならTransRベースの意味的注意機構を使い、0ならAKDN互換のrelation-aware注意機構を使う。
    parser.add_argument('--use_transr_attention', type=int, default=1, choices=[0, 1],
                        help='1: use TransR-based semantic attention, 0: use AKDN-compatible relation-aware attention.')
        # 1ならtauでスケーリングしたsoftmaxを使い、0なら通常のsoftmaxを使う（tauは無視される）。
    parser.add_argument('--use_tau_softmax', type=int, default=1, choices=[0, 1],
                        help='1: apply tau-scaled softmax, 0: use standard softmax (tau ignored).')
        # 1なら注意logitに距離ペナルティ分岐を加え、0なら意味スコアのみで計算する。
    parser.add_argument('--use_dist_penalty', type=int, required=True, choices=[0, 1],
                        help='1: add distance branch to attention logit, 0: semantic score only.')
        # 1なら距離係数にedge-wiseなGLU由来lambdaを使い、0なら固定係数1を使う（dist_penalty有効時のみ意味がある）。
    parser.add_argument('--use_glu_lambda', type=int, default=1, choices=[0, 1],
                        help='1: use edge-wise GLU lambda for dist coefficient, 0: use fixed coefficient 1. '
                             'Requires --use_dist_penalty=1 to have any effect.')
        # sem/dist 正規化方式を切り替える。
    parser.add_argument('--score_norm_mode', type=str, required=True,
                        choices=['neighbor_zscore', 'global_zscore', 'global_minmax'],
                        help='Normalization for sem/dist before GLU and attention fusion. '
                             'global_minmax computes min/max from all edge-wise raw scores in each forward pass.')
        # 1なら結合ベースの距離スコア、0なら負の距離を使う（TransR注意を使わない場合はAKDN埋め込み空間で距離計算）。
    parser.add_argument('--use_concat_dist', type=int, required=True, choices=[0, 1],
                        help='1: use concatenation-based dist score, 0: use negative distance. '
                             'When --use_transr_attention=0, the distance is computed in AKDN embedding space.')
        # 注意計算を分割してOOMを回避するためのチャンクサイズ。0は分割なし。
    parser.add_argument('--att_chunk_size', type=int, default=0,
                        help='Chunk size for attention computation to prevent OOM. 0 = no chunking (default). '
                             'Recommended: 262144 for Yelp2018.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--lambda_min', type=float, default=0.0,
                        help='Lower bound for GLU-generated edge-wise lambda.')
    parser.add_argument('--lambda_max', type=float, default=1.0,
                        help='Upper bound for GLU-generated edge-wise lambda.')
    parser.add_argument('--lambda_glu_hidden_dim', type=int, default=64,
                        help='Hidden size of the GLU MLP used to generate edge-wise lambda.')

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
