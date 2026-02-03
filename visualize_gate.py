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

    # 6. Plot Histograms
    logging.info("Plotting histograms...")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    def plot_histogram(data_tensor, title, xlabel, filename, color='blue'):
        # Flatten data
        data_np = data_tensor.view(-1).numpy()
        
        plt.figure(figsize=(10, 6))
        plt.hist(data_np, bins=50, alpha=0.75, color=color, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Save statistics
        mean_val = np.mean(data_np)
        std_val = np.std(data_np)
        plt.text(0.05, 0.95, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}', transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        save_path = os.path.join(args.save_dir, filename)
        plt.savefig(save_path)
        logging.info(f"Saved: {filename}")
        plt.close()

    # Static Weights (Shared across layers)
    wa_weights = model.W_a.weight.detach().cpu()
    wb_weights = model.W_b.weight.detach().cpu()
    
    # Layers to process: 1, 2, 3
    # Note: model.gate_* lists correspond to layers 0, 1, 2 index-wise
    num_layers = min(len(model.gate_coefficients), 3) # Should be 3 based on standard config
    
    # Data gathering for All Layers
    all_layers_data = {
        'wa': [], 'ekg': [], 'eig': [], 'wb': [], 
        'wa_ekg': [], 'wb_eig': [], 'sum': [], 'g': []
    }

    for i in range(num_layers):
        layer_num = i + 1
        
        # Layer specific data
        e_kg_layer = model.gate_kg[i]
        e_ig_layer = model.gate_ig[i]
        wa_ekg_layer = model.gate_wa_kg[i]
        wb_eig_layer = model.gate_wb_ig[i]
        sum_layer = model.gate_inputs[i]
        g_layer = model.gate_coefficients[i]
        
        # 1. w_a (Shared)
        plot_histogram(wa_weights, 
                       f'Layer {layer_num}: Distribution of $W_a$ Weights', 
                       'Weight Value', 
                       f'layer_{layer_num}_w_a.png', color='orange')
        all_layers_data['wa'].append(wa_weights) # Duplicate but conceptually consistent
        
        # 2. e_kg
        plot_histogram(e_kg_layer,
                       f'Layer {layer_num}: Distribution of $e_{{kg}}$',
                       'Embedding Value',
                       f'layer_{layer_num}_e_kg.png', color='green')
        all_layers_data['ekg'].append(e_kg_layer)

        # 3. e_ig
        plot_histogram(e_ig_layer,
                       f'Layer {layer_num}: Distribution of $e_{{ig}}$',
                       'Embedding Value',
                       f'layer_{layer_num}_e_ig.png', color='skyblue')
        all_layers_data['eig'].append(e_ig_layer)

        # 4. w_b (Shared)
        plot_histogram(wb_weights,
                       f'Layer {layer_num}: Distribution of $W_b$ Weights',
                       'Weight Value',
                       f'layer_{layer_num}_w_b.png', color='purple')
        all_layers_data['wb'].append(wb_weights)

        # 5. w_a * e_kg
        plot_histogram(wa_ekg_layer,
                       f'Layer {layer_num}: Distribution of $W_a e_{{kg}}$',
                       'Value',
                       f'layer_{layer_num}_wa_ekg.png', color='olive')
        all_layers_data['wa_ekg'].append(wa_ekg_layer)

        # 6. w_b * e_ig
        plot_histogram(wb_eig_layer,
                       f'Layer {layer_num}: Distribution of $W_b e_{{ig}}$',
                       'Value',
                       f'layer_{layer_num}_wb_eig.png', color='teal')
        all_layers_data['wb_eig'].append(wb_eig_layer)

        # 7. Sum (Pre-sigmoid)
        plot_histogram(sum_layer,
                       f'Layer {layer_num}: Distribution of $W_a e_{{kg}} + W_b e_{{ig}}$',
                       'Value',
                       f'layer_{layer_num}_sum.png', color='red')
        all_layers_data['sum'].append(sum_layer)

        # 8. g (Sigmoid output)
        plot_histogram(g_layer,
                       f'Layer {layer_num}: Distribution of Gate Coefficient $g$',
                       'Gate Value',
                       f'layer_{layer_num}_g.png', color='blue')
        all_layers_data['g'].append(g_layer)


    # Plot All Layers (Aggregated)
    logging.info("Plotting aggregated histograms...")
    
    # Helper for aggregated
    def plot_agg(key, title, filename, color):
        if not all_layers_data[key]: return
        combined = torch.cat(all_layers_data[key], dim=0) if isinstance(all_layers_data[key][0], torch.Tensor) else torch.cat([torch.tensor(x) for x in all_layers_data[key]], dim=0)
        plot_histogram(combined, title, 'Value', filename, color)

    plot_agg('wa', 'All Layers: Distribution of $W_a$ Weights', 'all_layers_w_a.png', 'orange')
    plot_agg('ekg', 'All Layers: Distribution of $e_{{kg}}$', 'all_layers_e_kg.png', 'green')
    plot_agg('eig', 'All Layers: Distribution of $e_{{ig}}$', 'all_layers_e_ig.png', 'skyblue')
    plot_agg('wb', 'All Layers: Distribution of $W_b$ Weights', 'all_layers_w_b.png', 'purple')
    plot_agg('wa_ekg', 'All Layers: Distribution of $W_a e_{{kg}}$', 'all_layers_wa_ekg.png', 'olive')
    plot_agg('wb_eig', 'All Layers: Distribution of $W_b e_{{ig}}$', 'all_layers_wb_eig.png', 'teal')
    plot_agg('sum', 'All Layers: Distribution of $W_a e_{{kg}} + W_b e_{{ig}}$', 'all_layers_sum.png', 'red')
    plot_agg('g', 'All Layers: Distribution of Gate Coefficient $g$', 'all_layers_g.png', 'blue')

if __name__ == "__main__":
    visualize_gate_coefficients()
