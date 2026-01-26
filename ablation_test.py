
import sys
import argparse
import os
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from parser.parser_akdn import parse_akdn_args
from data_loader.loader_akdn import DataLoaderAKDN
from model.AKDN import AKDN
from utils.model_helper import load_model
from utils.metrics import calc_metrics_at_k

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def custom_evaluate(model, data, Ks, device):
    """
    Evaluates the model and returns per-user metrics for analysis.
    """
    model.eval()
    
    test_batch_size = data.test_batch_size
    train_user_dict = data.train_user_dict
    test_user_dict = data.test_user_dict
    
    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    
    n_items = data.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)
    
    metric_names = ['recall', 'ndcg']
    # Storage for per-user metrics: {k: {'recall': [val, ...], 'ndcg': [val, ...]}}
    per_user_metrics = {k: {m: [] for m in metric_names} for k in Ks}
    all_evaluated_users = []

    with tqdm(total=len(user_ids_batches), desc='Evaluating') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)
            
            with torch.no_grad():
                batch_scores = model('calc_score', batch_user_ids, item_ids)
            
            batch_scores = batch_scores.cpu()
            
            # Calculate metrics
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)
            
            # Store results
            for k in Ks:
                for m in metric_names:
                    # metrics_at_k_batch returns an array of shape (batch_size,)
                    per_user_metrics[k][m].extend(batch_metrics[k][m])
            
            all_evaluated_users.extend(batch_user_ids.cpu().numpy())
            pbar.update(1)
            
    return np.array(all_evaluated_users), per_user_metrics

def analyze_groups(user_ids, per_user_metrics, train_user_dict, k=20):
    """
    Analyzes metrics by user group (interaction count).
    """
    # Calculate degree (interaction count) for each evaluated user
    user_degrees = [len(train_user_dict[u]) if u in train_user_dict else 0 for u in user_ids]
    user_degrees = np.array(user_degrees)
    
    # Define Groups based on interaction count
    # Example: Low (<15), Medium (15-50), High (>50)
    # Or Quartiles. Let's use the user's description flavor or standard buckets?
    # Let's try to detect reasonable buckets or use Fixed ones. 
    # Let's use fixed bins similar to common practice.
    
    groups = {
        'Sparse (<16)': (user_degrees < 16),
        'Medium (16-64)': (user_degrees >= 16) & (user_degrees < 64),
        'Dense (>=64)': (user_degrees >= 64)
    }
    
    results = {}
    
    recall_list = np.array(per_user_metrics[k]['recall'])
    ndcg_list = np.array(per_user_metrics[k]['ndcg'])
    
    for group_name, mask in groups.items():
        count = np.sum(mask)
        if count == 0:
            results[group_name] = {'count': 0, 'recall': 0.0, 'ndcg': 0.0}
            continue
            
        group_recall = recall_list[mask]
        group_ndcg = ndcg_list[mask]
        
        results[group_name] = {
            'count': count,
            'recall': np.mean(group_recall),
            'ndcg': np.mean(group_ndcg)
        }
        
    return results

def run_ablation():
    # Parse args (reuse parser logic)
    # To avoid sys.argv issues if running inside another env, we might need to handle args.
    # But usually we run this script directly.
    args = parse_akdn_args()
    
    # Ensuring Ks includes 20
    Ks = eval(args.Ks)
    if 20 not in Ks:
        Ks.append(20)
        args.Ks = str(Ks)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    logging.info("Loading Data...")
    data = DataLoaderAKDN(args, logging)
    
    # Initialize Model
    logging.info("Initializing Model...")
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    model = AKDN(args, data.n_users, data.n_entities, data.n_relations, 
                 A_in=data.norm_adj_mat, 
                 user_pre_embed=user_pre_embed, 
                 item_pre_embed=item_pre_embed,
                 edge_dropout_rate=0.0)
    
    # Load Weights
    if args.pretrain_model_path and os.path.exists(args.pretrain_model_path):
        logging.info(f"Loading weights from {args.pretrain_model_path}")
        model = load_model(model, args.pretrain_model_path)
    else:
        logging.warning("No pretrain_model_path provided. Running with random weights (Results will be meaningless if not trained).")

    model.to(device)
    
    # Set KG structure
    relations = list(data.train_relation_dict.keys())
    model.set_kg_structure(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device), relations)
    
    # ---------------------------------------------------------
    # Experiments
    # ---------------------------------------------------------
    experiments = [
        ('A', 'Normal', 'normal'),
        ('B', 'KG Blocked (g=0)', 'ig_only'),
        ('C', 'IG Blocked (g=1)', 'kg_only')
    ]
    
    final_results = []
    
    for exp_id, exp_name, gate_mode in experiments:
        logging.info(f"--- Running Experiment {exp_id}: {exp_name} ---")
        
        # Set Gate Control
        model.gate_control = gate_mode
        
        # Evaluate
        user_ids, metrics = custom_evaluate(model, data, Ks, device)
        
        # Overall Results for K=20
        k = 20
        overall_recall = np.mean(metrics[k]['recall'])
        overall_ndcg = np.mean(metrics[k]['ndcg'])
        
        logging.info(f"Exp {exp_id} Overall: Recall@{k} = {overall_recall:.4f}, NDCG@{k} = {overall_ndcg:.4f}")
        
        # Group Analysis
        group_res = analyze_groups(user_ids, metrics, data.train_user_dict, k=k)
        
        # Store for summary
        exp_res = {
            'Experiment': exp_name,
            'Overall Recall': overall_recall,
            'Overall NDCG': overall_ndcg
        }
        for g_name, g_vals in group_res.items():
            exp_res[f'{g_name} Recall'] = g_vals['recall']
            exp_res[f'{g_name} NDCG'] = g_vals['ndcg']
            
        final_results.append(exp_res)

        logging.info(f"Exp {exp_id} Group Analysis:")
        for g_name, g_vals in group_res.items():
            logging.info(f"  {g_name}: Recall={g_vals['recall']:.4f}, NDCG={g_vals['ndcg']:.4f}")
            
    # ---------------------------------------------------------
    # Summary & Visualization
    # ---------------------------------------------------------
    df = pd.DataFrame(final_results)
    print("\n=== Ablation Study Summary ===")
    print(df.to_string())
    
    # Save to CSV
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        csv_path = os.path.join(args.save_dir, 'ablation_results.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Results saved to {csv_path}")

        # Basic Plotting
        # Plot Recall Comparison
        try:
            metrics_to_plot = ['Overall Recall', 'Overall NDCG']
            # Also plot specific groups if they exist
            # ...
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(experiments))
            width = 0.35
            
            recalls = df['Overall Recall'].values
            ndcgs = df['Overall NDCG'].values
            
            rects1 = ax.bar(x - width/2, recalls, width, label='Recall@20')
            rects2 = ax.bar(x + width/2, ndcgs, width, label='NDCG@20')
            
            ax.set_ylabel('Score')
            ax.set_title('Ablation Study: Overall Performance')
            ax.set_xticks(x)
            ax.set_xticklabels([e[1] for e in experiments])
            ax.legend()
            
            plt.savefig(os.path.join(args.save_dir, 'ablation_overall_plot.png'))
            logging.info("Plot saved.")
        except Exception as e:
            logging.warning(f"Could not creat plot: {e}")

if __name__ == "__main__":
    run_ablation()
