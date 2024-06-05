import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from dataloaders.Graph_NMR_data import Graph_NMR_data, custom_collate_fn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import os
from models.GNN2d import GNNNodeEncoder
from models.Comenet import ComENet
from models.NMRModel import NodeEncodeInterface
from models.Schnet import SchNet
from train import train_model
from eval import eval_model
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
import pandas as pd

nmr_path = '/scratch0/yunruili/HSQC_data/all_annotation_data/nmr'
graph_path = '/scratch0/yunruili/HSQC_data/all_annotation_data/graph'
csv_file = '/scratch0/yunruili/HSQC_data/all_annotated_files.csv'

########### CHANGE
model_folder = './experiment_new'
save_file = 'eval_rslt_new.csv'

dataset = Graph_NMR_data(csv_file, graph_path, nmr_path)
# dataset = DataLoader(data, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)

torch.manual_seed(0)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Further split val_dataset into validation and test datasets
val_size = len(val_dataset) // 2
test_size = len(val_dataset) - val_size  # Ensure all data is used
val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])

# Optionally create DataLoader instances for each subset
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)


rslt = []
cols = ['batch_size', 'type', 'num_layers', 'hidden_channels', 'c_out_hidden', 'h_out_hidden', 'c_sol_emb_dim', 'h_sol_emb_dim',\
         'num_output_layers', 'num_filters', 'num_gaussians', 'closs', 'hloss']

model_list = os.listdir(model_folder)
for m in model_list:
    name = m.split('.')[0]
    comps = name.split('_')
    # model spec
    batch_size = int(comps[1][1:])
    type = comps[0]
    if type in ["gin", "gcn", "gat", "graphsage", "nnconv"]:
        num_layers = int(comps[7])
        hidden_channels = int(comps[5])
        c_out_hidden = [128, 64]
        h_out_hidden = [128, 64]
        c_sol_emb_dim = int(comps[13][:2])
        h_sol_emb_dim = int(comps[13][2:])
        num_output_layers = 0
        num_filters = 0
        num_gaussians = 0

        nodeEncoder = GNNNodeEncoder(int(num_layers), int(hidden_channels), JK="last", gnn_type=type, aggr='add')    
        ckpt_path = '%s_b%d_solventCH_%s_hiddendim_%d_nlayers_%d_couthidden_%s_houthidden_%s_solventdimch_%s.pt' % \
        (type, batch_size, 'sum', hidden_channels, num_layers, ''.join(str(i) for i in c_out_hidden), ''.join(str(i) for i in h_out_hidden), ''.join([str(c_sol_emb_dim), str(h_sol_emb_dim)]))
    

    # comenet
    elif type == 'comenet':
        # continue
        num_layers = int(comps[7])
        hidden_channels = int(comps[5])
        c_out_hidden = [128, 64]
        h_out_hidden = [128, 64]
        c_sol_emb_dim = int(comps[15][:2])
        h_sol_emb_dim = int(comps[15][2:])
        num_output_layers = int(comps[9])
        num_filters = 0
        num_gaussians = 0

        nodeEncoder = ComENet(in_embed_size=3, c_out_channels=1, h_out_channels=2, agg_method='sum', \
                    hidden_channels=hidden_channels, num_layers=num_layers, num_output_layers=num_output_layers)
        ckpt_path = '%s_b%d_solventCH_%s_hiddendim_%d_nlayers_%d_noutlayers_%d_couthidden_%s_houthidden_%s_solventdimch_%s.pt' % \
        (type, batch_size, 'sum', hidden_channels, num_layers, num_output_layers, ''.join(str(i) for i in c_out_hidden), ''.join(str(i) for i in h_out_hidden), ''.join([str(c_sol_emb_dim), str(h_sol_emb_dim)]))

       
    # schnet
    elif type == 'schnet':
        num_layers = int(comps[11])
        hidden_channels = int(comps[5])
        c_out_hidden = [128, 64]
        h_out_hidden = [128, 64]
        num_filters = int(comps[7])
        num_gaussians = int(comps[9])
        c_sol_emb_dim = int(comps[17][:2])
        h_sol_emb_dim = int(comps[17][2:])
        num_output_layers = 0

        nodeEncoder = SchNet(energy_and_force=False, cutoff=10.0, num_layers=num_layers, hidden_channels=hidden_channels, out_channels=1, num_filters=num_filters, num_gaussians=num_gaussians)
        ckpt_path = '%s_b%d_solventCH_%s_hiddendim_%d_nfilter_%d_ngaussian_%d_nlayers_%d_couthidden_%s_houthidden_%s_solventdimch_%s.pt' % \
        (type, batch_size, 'sum', hidden_channels, num_filters, num_gaussians, num_layers, ''.join(str(i) for i in c_out_hidden), ''.join(str(i) for i in h_out_hidden), ''.join([str(c_sol_emb_dim), str(h_sol_emb_dim)]))

    else:
        raise ValueError("Invalid graph convolution type.")

    model = NodeEncodeInterface(nodeEncoder, hidden_channels=hidden_channels, c_out_hidden=c_out_hidden, h_out_hidden=h_out_hidden, c_solvent_emb_dim = c_sol_emb_dim, h_solvent_emb_dim = h_sol_emb_dim, h_out_channels=2, use_solvent=True)

    model.load_state_dict(torch.load(os.path.join(model_folder, ckpt_path)))

    c_loss, h_loss = eval_model(model, test_loader)

    tmp = [batch_size, type, num_layers, hidden_channels, c_out_hidden, h_out_hidden, c_sol_emb_dim, h_sol_emb_dim, num_output_layers, num_filters, num_gaussians, c_loss.detach().cpu().numpy().item(), h_loss.detach().cpu().numpy().item()]
    rslt.append(tmp)

    print(ckpt_path)
    print(tmp)
    
    # break

rslt = pd.DataFrame(data=rslt, columns=cols)
rslt.to_csv(save_file)