# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
# from dataloaders.Graph_NMR_data import Graph_NMR_data, custom_collate_f
from load_huggingface_data import load_data_from_huggingface
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader
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
from models.gnn_transformer import GNNTransformer
from train import train_model
from eval import eval_model


def main(args):

    dataset = load_data_from_huggingface('train')
    # Split the graph_list into train and test sets (80/20 split)
    seed = args.seed
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=seed)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    print(len(train_data), len(val_data))

    if args.notransformer:

        if args.type in ["gin", "gcn", "gat", "graphsage", "nnconv"]:
            nodeEncoder = GNNNodeEncoder(args.num_layers, args.hidden_channels, JK="last", gnn_type=args.type, aggr='add')
            ckpt_path = '%s_b%d_solventCH_%s_hiddendim_%d_nlayers_%d_couthidden_%s_houthidden_%s_solventdimch_%s_seed_%s.pt' % \
            (args.type, args.batch_size, args.agg_method, args.hidden_channels, args.num_layers, ''.join(str(i) for i in args.c_out_hidden), ''.join(str(i) for i in args.h_out_hidden), ''.join([str(args.c_sol_emb_dim), str(args.h_sol_emb_dim)]), args.seed)
        
        elif args.type == 'comenet':
            nodeEncoder = ComENet(in_embed_size=3, c_out_channels=1, h_out_channels=2, agg_method='sum', \
                        hidden_channels=args.hidden_channels, num_layers=args.num_layers, num_output_layers=args.num_output_layers)
            ckpt_path = '%s_b%d_solventCH_%s_hiddendim_%d_nlayers_%d_noutlayers_%d_couthidden_%s_houthidden_%s_solventdimch_%s_seed_%s.pt' % \
            (args.type, args.batch_size, args.agg_method, args.hidden_channels, args.num_layers, args.num_output_layers, ''.join(str(i) for i in args.c_out_hidden), ''.join(str(i) for i in args.h_out_hidden), ''.join([str(args.c_sol_emb_dim), str(args.h_sol_emb_dim)]), args.seed)
        
        elif args.type == 'schnet':
            nodeEncoder = SchNet(energy_and_force=False, cutoff=10.0, num_layers=args.num_layers, hidden_channels=args.hidden_channels, out_channels=1, num_filters=args.num_filters, num_gaussians=args.num_gaussians)
            ckpt_path = '%s_b%d_solventCH_%s_hiddendim_%d_nfilter_%d_ngaussian_%d_nlayers_%d_couthidden_%s_houthidden_%s_solventdimch_%s_seed_%s.pt' % \
            (args.type, args.batch_size, args.agg_method, args.hidden_channels, args.num_filters, args.num_gaussians, args.num_layers, ''.join(str(i) for i in args.c_out_hidden), ''.join(str(i) for i in args.h_out_hidden), ''.join([str(args.c_sol_emb_dim), str(args.h_sol_emb_dim)]), args.seed)

        else:
            raise ValueError("Invalid graph convolution type.")
        
        model = NodeEncodeInterface(nodeEncoder, hidden_channels=args.hidden_channels, c_out_hidden=args.c_out_hidden, h_out_hidden=args.h_out_hidden, c_solvent_emb_dim = args.c_sol_emb_dim, h_solvent_emb_dim = args.h_sol_emb_dim, h_out_channels=2, use_solvent=args.use_solvent)

    else:

        nodeEncoder = GNNTransformer(args)
        ckpt_path = 'trans_%s_b%d_solventCH_%s_hiddendim_%d_ngnn_%d_dmodel_%d_nhead_%d_dff_%d_ntrans_%d_couthidden_%s_houthidden_%s_solventdimch_%s_seed_%s.pt' % \
            (args.type, args.batch_size, args.agg_method, args.hidden_channels, args.num_layers,\
             args.d_model, args.nhead, args.dim_feedforward,args.num_encoder_layers, \
             ''.join(str(i) for i in args.c_out_hidden), ''.join(str(i) for i in args.h_out_hidden), ''.join([str(args.c_sol_emb_dim), str(args.h_sol_emb_dim)]), args.seed)
        
    
        model = NodeEncodeInterface(nodeEncoder, hidden_channels=args.d_model, c_out_hidden=args.c_out_hidden, h_out_hidden=args.h_out_hidden, c_solvent_emb_dim = args.c_sol_emb_dim, h_solvent_emb_dim = args.h_sol_emb_dim, h_out_channels=2, use_solvent=args.use_solvent)

    print(model)

    count_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameter: %d" % count_param)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)

    print(ckpt_path)
    print( 'final_%s'%ckpt_path)

    model = train_model(model, dataloaders, optimizer, exp_lr_scheduler, ckpt_path, num_epochs=args.n_epoch)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    
    args.add_argument('--batch_size', type=int, default=32, help='batch size')
    args.add_argument('--n_epoch', type=int, default=150, help='num of epoches')
    args.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    args.add_argument('--type', type=str, default='gin', help='GNN type')
    args.add_argument('--notransformer', action='store_true', help='Disable transformer')
    args.add_argument('--hidden_channels', type=int, default=256, help='hidden channel of gnn')
    args.add_argument('--num_layers', type=int, default=2, help='number of layers for GNN')
    # args.add_argument('--num_output_layers', type=int, default=2, help='number of layers for GNN')
    args.add_argument('--agg_method', type=str, default='sum', help='aggregation method for GNN')
    args.add_argument('--c_out_hidden', default=[128, 64], type=int, nargs="+", help='hidden dims of projection')
    args.add_argument('--h_out_hidden', default=[128, 64], type=int, nargs="+", help='hidden dims of projection')
    args.add_argument('--c_sol_emb_dim', type=int, default=16, help='carbon solvent embedding dimension')
    args.add_argument('--h_sol_emb_dim', type=int, default=32, help='hydrogen solvent embedding dimension')
    # comenet
    args.add_argument('--num_output_layers', type=int, default=2, help='number of layers for GNN')
    # schenet
    args.add_argument('--num_gaussians', type=int, default=50, help='number of gaussians for dist embedding and edge update')
    args.add_argument('--num_filters', type=int, default=128, help='number of channels for edge update')

    # transformer args
    args.add_argument('--max_seq_len', type=int, default=None, help='maximum sequence length to predict (default: None)')
    args.add_argument("--pos_encoder", default=False, action='store_true')
    args.add_argument("--pretrained_gnn", type=str, default=None, help="pretrained gnn_node node embedding path")
    args.add_argument("--freeze_gnn", type=int, default=None, help="Freeze gnn_node weight from epoch `freeze_gnn`")
    args.add_argument("--d_model", type=int, default=128, help="transformer d_model.")
    args.add_argument("--nhead", type=int, default=4, help="transformer heads")
    args.add_argument("--dim_feedforward", type=int, default=512, help="transformer feedforward dim")
    args.add_argument("--transformer_dropout", type=float, default=0.3)
    args.add_argument("--transformer_activation", type=str, default="relu")
    args.add_argument("--num_encoder_layers", type=int, default=4)
    args.add_argument("--max_input_len", default=1000, help="The max input length of transformer input")
    args.add_argument("--transformer_norm_input", action="store_true", default=False)
    args.add_argument('--dropout', type=float, default=0.3, help='Dropout')   

    
    args = args.parse_args()

    args.use_solvent = True

    for seed in [42]:
        args.seed = seed
        main(args)
# %%
