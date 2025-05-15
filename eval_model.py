import torch
from load_huggingface_data import load_data_from_huggingface
from torch_geometric.data import DataLoader
import os
from models.GNN2d import GNNNodeEncoder
from models.gnn_transformer import GNNTransformer
from models.Comenet import ComENet
from models.NMRModel import NodeEncodeInterface
from models.Schnet import SchNet
from eval import eval_model
import pandas as pd
import argparse

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

split = 'eval'
dataset = load_data_from_huggingface(split=split)
dataset = DataLoader(dataset, batch_size=1, shuffle=False)

model_folder = './experiment/comenet'
rslt = []

# trans_gin_b32_solventCH_sum_hiddendim_256_ngnn_2_dmodel_128_nhead_2_dff_256_ntrans_3_couthidden_25664_houthidden_25664_solventdimch_1616_seed_0.pt
#        1   2                            6       8       10         12    14        16           18                20                   22     24

model_list = os.listdir(model_folder)
for m in model_list:
    if not m.endswith('.pt'):
        continue
    name = m.split('.')[0]
    comps = name.split('_') 
    # model spec
    if comps[0] == 'trans':
        save_file = 'eval_rslt_trans.csv'
        cols = ['batch_size', 'type', 'num_layers', 'hidden_channels', 'c_out_hidden', 'h_out_hidden', 'c_sol_emb_dim', 'h_sol_emb_dim',\
         'd_model', 'nhead', 'd_ff', 'num_trans_layers', 'seed', 'closs', 'hloss']
        batch_size = int(comps[2][1:])
        type = comps[1]
        if type in ["gin", "gcn", "gat", "graphsage", "nnconv"]:
            num_layers = int(comps[8])
            hidden_channels = int(comps[6])
            d_model = int(comps[10])
            nhead = int(comps[12])
            dff = int(comps[14])
            num_trans_layers = int(comps[16])
            c_out_hidden = [int(comps[18][:3]), int(comps[18][3:])]
            h_out_hidden = [int(comps[20][:3]), int(comps[20][3:])]
            c_sol_emb_dim = int(comps[22][:2])
            h_sol_emb_dim = int(comps[22][2:])
            seed = int(comps[24])
            
            args.type = type
            args.num_layers = num_layers
            args.hidden_channels = hidden_channels
            args.d_model = d_model
            args.nhead = nhead
            args.dim_feedforward = dff
            args.num_encoder_layers = num_trans_layers
            args.c_out_hidden = c_out_hidden
            args.h_out_hidden = h_out_hidden
            args.c_sol_emb_dim = c_sol_emb_dim
            args.h_sol_emb_dim = h_sol_emb_dim
            args.seed = seed

            nodeEncoder = GNNTransformer(args)
            ckpt_path = 'trans_%s_b%d_solventCH_%s_hiddendim_%d_ngnn_%d_dmodel_%d_nhead_%d_dff_%d_ntrans_%d_couthidden_%s_houthidden_%s_solventdimch_%s_seed_%s.pt' % \
                (args.type, args.batch_size, args.agg_method, args.hidden_channels, args.num_layers,\
                args.d_model, args.nhead, args.dim_feedforward,args.num_encoder_layers, \
                ''.join(str(i) for i in args.c_out_hidden), ''.join(str(i) for i in args.h_out_hidden), ''.join([str(args.c_sol_emb_dim), str(args.h_sol_emb_dim)]), args.seed)
    
            model = NodeEncodeInterface(nodeEncoder, hidden_channels=args.d_model, c_out_hidden=args.c_out_hidden, h_out_hidden=args.h_out_hidden, c_solvent_emb_dim = args.c_sol_emb_dim, h_solvent_emb_dim = args.h_sol_emb_dim, h_out_channels=2, use_solvent=args.use_solvent)
            model.load_state_dict(torch.load(os.path.join(model_folder, ckpt_path)))

            c_loss, h_loss = eval_model(model, dataset)

            tmp = [batch_size, type, num_layers, hidden_channels, c_out_hidden, h_out_hidden, c_sol_emb_dim, h_sol_emb_dim, 
                 d_model, nhead, dff, num_trans_layers, seed, c_loss.detach().cpu().numpy().item(), h_loss.detach().cpu().numpy().item()]
            rslt.append(tmp)

            print(ckpt_path)
            print(tmp)

    else: 
        save_file = 'eval_rslt_comenet_%s.csv'%split
        cols = ['batch_size', 'type', 'num_layers', 'hidden_channels', 'c_out_hidden', 'h_out_hidden', 'c_sol_emb_dim', 'h_sol_emb_dim',\
         'num_output_layers', 'num_filters', 'num_gaussians', 'seed', 'closs', 'hloss']
        
        batch_size = int(comps[1][1:])
        type = comps[0]
        if type in ["gin", "gcn", "gat", "graphsage", "nnconv"]:
            num_layers = int(comps[7])
            hidden_channels = int(comps[5])
            c_out_hidden = [int(comps[9][:3]), int(comps[9][3:])]
            h_out_hidden = [int(comps[11][:3]), int(comps[11][3:])]
            c_sol_emb_dim = int(comps[13][:2])
            h_sol_emb_dim = int(comps[13][2:])
            seed = int(comps[15])
            num_output_layers = 0
            num_filters = 0
            num_gaussians = 0

            nodeEncoder = GNNNodeEncoder(int(num_layers), int(hidden_channels), JK="last", gnn_type=type, aggr='add')    
            ckpt_path = '%s_b%d_solventCH_%s_hiddendim_%d_nlayers_%d_couthidden_%s_houthidden_%s_solventdimch_%s_seed_%s.pt' % \
            (type, batch_size, 'sum', hidden_channels, num_layers, ''.join(str(i) for i in c_out_hidden), ''.join(str(i) for i in h_out_hidden), ''.join([str(c_sol_emb_dim), str(h_sol_emb_dim)]), seed)
    

        # comenet
        elif type == 'comenet':
            # continue
            num_layers = int(comps[7])
            hidden_channels = int(comps[5])
            c_out_hidden = [int(comps[11][:3]), int(comps[11][3:])]
            h_out_hidden = [int(comps[13][:3]), int(comps[13][3:])]
            c_sol_emb_dim = int(comps[15][:2])
            h_sol_emb_dim = int(comps[15][2:])
            num_output_layers = int(comps[9])
            num_filters = 0
            num_gaussians = 0
            seed = int(comps[17])

            nodeEncoder = ComENet(in_embed_size=3, c_out_channels=1, h_out_channels=2, agg_method='sum', \
                        hidden_channels=hidden_channels, num_layers=num_layers, num_output_layers=num_output_layers)
            ckpt_path = '%s_b%d_solventCH_%s_hiddendim_%d_nlayers_%d_noutlayers_%d_couthidden_%s_houthidden_%s_solventdimch_%s_seed_%s.pt' % \
            (type, batch_size, 'sum', hidden_channels, num_layers, num_output_layers, ''.join(str(i) for i in c_out_hidden), ''.join(str(i) for i in h_out_hidden), ''.join([str(c_sol_emb_dim), str(h_sol_emb_dim)]), seed)

        
        # schnet
        elif type == 'schnet':
            num_layers = int(comps[11])
            hidden_channels = int(comps[5])
            c_out_hidden = [int(comps[13][:3]), int(comps[13][3:])]
            h_out_hidden = [int(comps[15][:3]), int(comps[15][3:])]
            num_filters = int(comps[7])
            num_gaussians = int(comps[9])
            c_sol_emb_dim = int(comps[17][:2])
            h_sol_emb_dim = int(comps[17][2:])
            num_output_layers = 0
            seed = comps[19]

            nodeEncoder = SchNet(energy_and_force=False, cutoff=10.0, num_layers=num_layers, hidden_channels=hidden_channels, out_channels=1, num_filters=num_filters, num_gaussians=num_gaussians)
            ckpt_path = '%s_b%d_solventCH_%s_hiddendim_%d_nfilter_%d_ngaussian_%d_nlayers_%d_couthidden_%s_houthidden_%s_solventdimch_%s_seed_%s.pt' % \
            (type, batch_size, 'sum', hidden_channels, num_filters, num_gaussians, num_layers, ''.join(str(i) for i in c_out_hidden), ''.join(str(i) for i in h_out_hidden), ''.join([str(c_sol_emb_dim), str(h_sol_emb_dim)]), seed)

        else:
            raise ValueError("Invalid graph convolution type.")

        model = NodeEncodeInterface(nodeEncoder, hidden_channels=hidden_channels, c_out_hidden=c_out_hidden, h_out_hidden=h_out_hidden, c_solvent_emb_dim = c_sol_emb_dim, h_solvent_emb_dim = h_sol_emb_dim, h_out_channels=2, use_solvent=True)

        model.load_state_dict(torch.load(os.path.join(model_folder, ckpt_path)))

        c_loss, h_loss = eval_model(model, dataset)

        tmp = [batch_size, type, num_layers, hidden_channels, c_out_hidden, h_out_hidden, c_sol_emb_dim, h_sol_emb_dim, num_output_layers, num_filters, num_gaussians, seed, c_loss.detach().cpu().numpy().item(), h_loss.detach().cpu().numpy().item()]
        rslt.append(tmp)

        print(ckpt_path)
        print(tmp)
    
    # break

rslt = pd.DataFrame(data=rslt, columns=cols)
rslt.to_csv(save_file)