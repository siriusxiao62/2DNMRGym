from datasets import load_dataset
from torch_geometric.data import Data, DataLoader
import torch

def load_data_from_huggingface(split='train'):
    data = load_dataset('siriusxiao/2DNMRGym', split=split)
    # Iterate over each sample in the dataset
    graph_list = []  # To hold PyG Data objects

    for sample in data:
        # Extract graph-related data from the sample
        graph_data = sample['graph_data']  # The graph data is in the form of a dictionary
        c_peaks = sample['c_peaks']
        h_peaks = sample['h_peaks']
        filename = sample['filename']

        # Convert graph data to PyG Data object
        pyg_data = Data(
            x=torch.tensor(graph_data['x'], dtype=torch.float),  # Node features
            edge_index=torch.tensor(graph_data['edge_index'], dtype=torch.long),  # Edge indices
            edge_attr=torch.tensor(graph_data['edge_attr'], dtype=torch.long) if graph_data['edge_attr'] else None,  # Edge attributes (if present)
            pos=torch.tensor(graph_data['pos'], dtype=torch.float) if graph_data['pos'] else None,  # Node positions (if present)
            solvent_class=torch.tensor([graph_data['solvent_class']], dtype=torch.long),  # Solvent class (single value)
            cnmr=torch.tensor(c_peaks, dtype=torch.float),  # Chemical shift (C)
            hnmr=torch.tensor(h_peaks, dtype=torch.float),  # Chemical shift (H)
            filename=filename  # Filename (optional)
        )
        
        # Append the PyG Data object to the list
        graph_list.append(pyg_data)
    
    return graph_list
