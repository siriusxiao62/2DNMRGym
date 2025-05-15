import torch
import torch.nn as nn
import time
from rdkit import Chem
from gen_graph_from_smiles import smiles_to_pyg_graph_2d, smiles_to_pyg_graph_3d
from torch_geometric.data import Data, Batch
import torch.nn.functional as F



def eval_model(model, dataloader):
    model = model.cuda()

    since = time.time()
    model.eval()

    total_loss_c = 0
    total_loss_h = 0
    print(len(dataloader))
    with torch.no_grad():
        for graph in dataloader:
            graph = graph.cuda()

            with torch.cuda.amp.autocast():
                try:
                    [c_shifts, h_shifts], c_idx = model(graph)
                except:
                    continue
 
                c_loss = F.l1_loss(c_shifts, graph.cnmr) * 200
                h_loss = F.l1_loss(h_shifts, graph.hnmr) * 10
                total_loss_c += c_loss
                total_loss_h += h_loss
            # print(loss)
                
        total_loss_c = total_loss_c / (len(dataloader))
        total_loss_h = total_loss_h / (len(dataloader))

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Val C loss: {:4f}, H loss: {:4f}'.format(total_loss_c, total_loss_h))

    return total_loss_c, total_loss_h

def eval_one_molecule(model, smile, solvent, type='2d'):
    mol = Chem.MolFromSmiles(smile)
    if type=='2d':
        mol_graph = smiles_to_pyg_graph_2d(mol)
    else:
        mol_graph = smiles_to_pyg_graph_2d(mol, max_attempts=5)
    mol_graph.has_c = True
    mol_graph.has_h = True
    mol_graph.batch = torch.zeros([len(mol_graph.x)], dtype=int)
    mol_graph.solvent_class = solvent
    mol_graph_list = [mol_graph]  # Since it's just one molecule, we wrap it in a list
    mol_graph_batch = Batch.from_data_list(mol_graph_list)
    # make prediction
    [c_shifts, h_shifts], ch_idx = model(mol_graph_batch)
    # cnmr = gt[['C']].values()
    # hnmr = gt[['H1', 'H2']].values()
    # loss = nn.MSELoss()(c_shifts, cnmr) + nn.MSELoss()(h_shifts, hnmr)
    return c_shifts, h_shifts, ch_idx

    