# %%
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


def train_model(model, dataloaders, optimizer, scheduler, checkpoint_path, num_epochs=1):
    best_loss = 1e10

    # model = model.cuda()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0

            print(len(dataloaders[phase]))
            for batch in dataloaders[phase]:
                graph, cnmr, hnmr, filename = batch
                # print(filename)
                # graph = graph.cuda()
                # cnmr = cnmr.cuda()
                # hnmr = hnmr.cuda()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    [c_shifts, h_shifts], c_idx = model(graph)
                    # c_nodes = (graph.x[:,0]==5).nonzero(as_tuple=True)[0]
                    # h_nodes = (graph.x[:, 0] == 0).nonzero(as_tuple=True)[0] 

                    # ##### calculate the indices of C node connected to H
                    # # Initialize a list to store C nodes connected to H
                    # c_nodes_connected_to_h = []
                    # # Check each C node for connection to any H node
                    # for c_node in c_nodes:
                    #     # Get indices of edges involving the C node
                    #     edges_of_c = (graph.edge_index[0] == c_node) | (graph.edge_index[1] == c_node)

                    #     # Get all nodes that are connected to this C node
                    #     connected_nodes = torch.cat((graph.edge_index[0][edges_of_c], graph.edge_index[1][edges_of_c])).unique()

                    #     # Check if any of these connected nodes are H nodes
                    #     if any(node in h_nodes for node in connected_nodes):
                    #         c_nodes_connected_to_h.append(c_node.item())
                    # # Convert to a tensor
                    # c_nodes_connected_to_h = torch.tensor(c_nodes_connected_to_h).cuda()
                    # c_index = [i for i, x in enumerate(c_nodes) if x in c_nodes_connected_to_h]
                    # c_shifts = c_shifts[c_index, :]
                    # # h_shifts = h_shifts[c_index, :]
                    
                    loss = nn.MSELoss()(c_shifts, cnmr) + nn.MSELoss()(h_shifts, hnmr)
                    loss *= 100
                    epoch_loss += loss
                    # print(loss)
                    if torch.isnan(loss):
                        print(filename)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
            epoch_loss = epoch_loss / (len(dataloaders[phase]))
            print(phase + 'loss', epoch_loss)
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

            # save the model weights
            if phase == 'val' and epoch_loss < best_loss:
                print(f"saving best model to {checkpoint_path}")
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint_path)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # save the last trained model
    torch.save(model.state_dict(), 'final_%s'%checkpoint_path)

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def main(args):

    nmr_path = '/Users/siriusxiao/Documents/Github/2dNMR_annotation_data/HSQC_data/good_annotation/nmr'
    graph_path = '/Users/siriusxiao/Documents/Github/2dNMR_annotation_data/HSQC_data/good_annotation/graph'
    csv_file = '/Users/siriusxiao/Documents/Github/2dNMR_annotation_data/HSQC_data/good_annotation.csv'

    dataset = Graph_NMR_data(csv_file, graph_path, nmr_path)
    # dataset = DataLoader(data, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)

    torch.manual_seed(0)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    if args.type in ["gin", "gcn", "gat", "graphsage", "nnconv"]:
        nodeEncoder = GNNNodeEncoder(args.num_layers, args.hidden_channels, JK="last", gnn_type=args.type, aggr='add')
    elif args.type == 'comenet':
        nodeEncoder = ComENet(in_embed_size=3, c_out_channels=1, h_out_channels=2, agg_method='sum', \
                    hidden_channels=args.hidden_channels, num_layers=args.num_layers, num_output_layers=args.num_output_layers)
    else:
        raise ValueError("Invalid graph convolution type.")
    
    model = NodeEncodeInterface(nodeEncoder, hidden_channels=args.hidden_channels, c_out_hidden=args.c_out_hidden, h_out_hidden=args.h_out_hidden, c_solvent_emb_dim = args.c_sol_emb_dim, h_solvent_emb_dim = args.h_sol_emb_dim, h_out_channels=2, use_solvent=args.use_solvent)

    print(model)

    count_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameter: %d" % count_param)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Layer: {name}, number of params: {param.numel()}")

    # Fine-tuning setup
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.9)

    ckpt_path = '%s_solventCH_%s_hiddendim_%d_nlayers_%d_couthidden_%s_houthidden_%s_solventdimch_%s.pt' % \
        (args.type, args.agg_method, args.hidden_channels, args.num_layers, ''.join(str(i) for i in args.c_out_hidden), ''.join(str(i) for i in args.h_out_hidden), ''.join([str(args.c_sol_emb_dim), str(args.h_sol_emb_dim)]))
    
    print(ckpt_path)
    print( 'final_%s'%ckpt_path)

    # if os.path.exists(ckpt_path):
    #     msg = model.load_state_dict(torch.load(ckpt_path))
    #     print(msg)
    #     print('model loaded')
    model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, ckpt_path, num_epochs=args.n_epoch)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=32, help='batch size')
    args.add_argument('--n_epoch', type=int, default=40, help='num of epoches')
    args.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    args.add_argument('--type', type=str, default='comenet', help='GNN type')
    args.add_argument('--hidden_channels', type=int, default=512, help='hidden channel of gnn')
    args.add_argument('--num_layers', type=int, default=5, help='number of layers for GNN')
    # args.add_argument('--num_output_layers', type=int, default=2, help='number of layers for GNN')
    args.add_argument('--agg_method', type=str, default='sum', help='aggregation method for GNN')
    args.add_argument('--c_out_hidden', default=[128, 64], type=int, nargs="+", help='hidden dims of projection')
    args.add_argument('--h_out_hidden', default=[128, 64], type=int, nargs="+", help='hidden dims of projection')
    args.add_argument('--c_sol_emb_dim', type=int, default=16, help='carbon solvent embedding dimension')
    args.add_argument('--h_sol_emb_dim', type=int, default=32, help='hydrogen solvent embedding dimension')
    # comenet
    args.add_argument('--num_output_layers', type=int, default=2, help='number of layers for GNN')
    

    
    args = args.parse_args()

    args.use_solvent = True

    main(args)
# %%
