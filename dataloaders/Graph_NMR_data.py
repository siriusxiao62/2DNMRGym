import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
import scipy
from torch_geometric.data import Batch

class Graph_NMR_data(Dataset):
    '''
    Returns 
        graph data as usual
        c_peaks of shape [N, 1]
        h_peaks of shape [N, 2]
        filename as usual
    '''
    def __init__(self, csv_file, graph_path, nmr_path):
        df = pd.read_csv(csv_file)
        self.file_list = df['File_name'].to_list()
        self.solvent_class = df['solvent_class'].to_list()
        self.nmr_path = nmr_path
        self.graph_path = graph_path
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, item):
        filename = self.file_list[item].split('.')[0]
        solvent_class = torch.tensor(self.solvent_class[item])
        graph_file = os.path.join(self.graph_path, filename + '.pickle')
        graph_data = pickle.load(open(graph_file, 'rb'))
        graph_data.x = graph_data.x.float()
        graph_data.solvent_class = solvent_class

        # use processed file to load c and h peaks
        nmr = os.path.join(self.nmr_path, filename + '.csv')
        nmr_data = pd.read_csv(nmr)

        c_peaks = torch.tensor(nmr_data['C'].values).view(-1, 1)/200.0
        h_peaks = torch.tensor(nmr_data[['H1', 'H2']].values)/10.0

        return graph_data, c_peaks.float(), h_peaks.float(), filename
    
def custom_collate_fn(batch):
    # Separate graph data, NMR data, and filenames
    graphs, c_peaks, h_peaks, filenames = zip(*batch)

    # Use torch_geometric's Batch to handle graph data
    batched_graph = Batch.from_data_list(graphs)

    # Concatenate NMR data into a single tensor
    batched_cnmr_data = torch.cat([data for data in c_peaks], dim=0)
    batched_hnmr_data = torch.cat([data for data in h_peaks], dim=0)

    return batched_graph, batched_cnmr_data, batched_hnmr_data, filenames


# nmr_path = '/Users/siriusxiao/Documents/Github/2dNMR_annotation_data/HSQC_data/good_annotation/nmr'
# graph_path = '/Users/siriusxiao/Documents/Github/2dNMR_annotation_data/HSQC_data/good_annotation/graph'
# csv_file = '/Users/siriusxiao/Documents/Github/2dNMR_annotation_data/HSQC_data/good_annotation.csv'

# data = Graph_NMR_data(csv_file, graph_path, nmr_path)
# dataset = DataLoader(data, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
# for graph_data, cnmr_data, hnmr_data, filename in dataset:
#     # print(graph_data.pos.shape)
#     # print(graph_data.x.shape)
#     print(cnmr_data.shape)
#     print(hnmr_data.shape)
#     print(cnmr_data)
#     print(hnmr_data)
#     break