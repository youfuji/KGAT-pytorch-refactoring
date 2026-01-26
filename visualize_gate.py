import sys
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from parser.parser_akdn import parse_akdn_args
from data_loader.loader_akdn import DataLoaderAKDN
from model.AKDN import AKDN
from utils.model_helper import load_model
from main_akdn import evaluate

def visualize_gate_coefficients():
    # Pre-parse arguments to handle script-specific args and remove them from sys.argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--model_type', type=str, default=None)
    
    local_args, remaining_argv = parser.parse_known_args()
    
    # Update sys.argv so parse_akdn_args doesn't complain
    sys.argv = [sys.argv[0]] + remaining_argv

    # 1. Parse Arguments
    args = parse_akdn_args()
    
    # Override save_dir if provided
    if local_args.save_dir:
        args.save_dir = local_args.save_dir
    
    # Setup simple logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 2. Load Data
    logging.info("Loading data...")
    data = DataLoaderAKDN(args, logging)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Initialize Model
    logging.info("Initializing model...")
    
    # Pretrained Embeddings handling
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

    # 4. Load Trained Weights
    if args.pretrain_model_path and os.path.exists(args.pretrain_model_path):
        logging.info(f"Loading model from {args.pretrain_model_path}")
        model = load_model(model, args.pretrain_model_path)
    else:
        logging.warning("No pretrain_model_path specified or file not found. Using random/initial weights. (Is this intended?)")
        
    model.to(device)
    model.eval()
    
    # Set KG structure
    relations = list(data.train_relation_dict.keys())
    model.set_kg_structure(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device), relations)

    # Calculate and log Metrics (Recall@20, NDCG@20)
    # Ensure Ks includes 20
    Ks = eval(args.Ks)
    if 20 not in Ks:
        Ks.append(20)
        args.Ks = str(Ks) 
    
    logging.info(f"Evaluating model performance (Recall, NDCG @ {Ks})...")
    _, metrics_dict = evaluate(model, data, Ks, device)
    
    k = 20
    if k in metrics_dict:
        recall = metrics_dict[k]['recall']
        ndcg = metrics_dict[k]['ndcg']
        logging.info(f"Performance: Recall@{k} = {recall:.4f}, NDCG@{k} = {ndcg:.4f}")
    else:
        logging.warning(f"Metrics for K={k} not found in results.")


    # 5. Extract Gate Coefficients
    logging.info("Extracting gate coefficients...")
    model.record_gate = True
    
    with torch.no_grad():
        model.get_embeddings()
    
    gate_coeffs = model.gate_coefficients
    
    if not gate_coeffs:
        logging.error("No gate coefficients recorded.")
        return

    # 6. Plot Histogram
    logging.info("Plotting histogram...")
    
    # Aggregate all values from all layers
    all_gates = torch.cat(gate_coeffs, dim=0).view(-1).numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title('Distribution of Gate Coefficient $g$ (Fusion Gate)')
    plt.xlabel('Gate Value $g$ (0: Use IG, 1: Use KG)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Save statistics
    mean_g = np.mean(all_gates)
    std_g = np.std(all_gates)
    plt.text(0.05, 0.95, f'Mean: {mean_g:.4f}\nStd: {std_g:.4f}', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_path = os.path.join(args.save_dir, 'gate_coefficient_histogram.png')
    plt.savefig(save_path)
    logging.info(f"Histogram saved to {save_path}")
    
    # Optional: Plot per layer
    if len(gate_coeffs) > 1:
        plt.figure(figsize=(10, 6))
        for i, g_layer in enumerate(gate_coeffs):
            g_np = g_layer.view(-1).numpy()
            plt.hist(g_np, bins=50, alpha=0.5, label=f'Layer {i+1}', density=True)
        
        plt.title('Distribution of Gate Coefficient $g$ (Per Layer)')
        plt.xlabel('Gate Value $g$')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path_layer = os.path.join(args.save_dir, 'gate_coefficient_histogram_per_layer.png')
        plt.savefig(save_path_layer)
        logging.info(f"Per-layer histogram saved to {save_path_layer}")

if __name__ == "__main__":
    visualize_gate_coefficients()
