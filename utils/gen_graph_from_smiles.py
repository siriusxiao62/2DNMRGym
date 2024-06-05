import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import numpy as np
import os
import pandas as pd
import pickle
import argparse

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)), ### changed
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_atom_partial_charge': (-1.00000, 1.00000),
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def smiles_to_pyg_graph_3d(smiles, max_attempts=5):
    # Convert the SMILES string to an RDKit molecule
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string")

    # Add hydrogens
    molecule = Chem.AddHs(molecule)

    # Generate 3D coordinates
    # Attempt to generate 3D coordinates up to max_attempts times
    for attempt in range(max_attempts):
        if AllChem.EmbedMolecule(molecule, AllChem.ETKDG()) == 0:
            break
    AllChem.UFFOptimizeMolecule(molecule)

    # Get atom features (atomic number, chirality, hybridization)
    atom_features_list = []
    for atom in molecule.GetAtoms():
        atom_feature = \
        [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
        [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())] + \
        [allowable_features['possible_hybridization_list'].index(atom.GetHybridization())]

        atom_features_list.append(atom_feature)
        #print(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # Get atom positions (coordinates)
    conformer = molecule.GetConformer()
    pos = torch.tensor([list(conformer.GetAtomPosition(i)) for i in range(molecule.GetNumAtoms())], dtype=torch.float)

    # Get edge indices
    num_bond_features = 2
    if len(molecule.GetBonds()) > 0:
        edge_list = []
        edge_feature_list = []
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                    bond.GetBondType())] + [allowable_features['possible_bond_dirs'].index(
                    bond.GetBondDir())]
            edge_list.append((i, j))
            edge_feature_list.append(edge_feature)
            edge_list.append((j, i))  # add both directions for undirected graph
            edge_feature_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_feature_list), dtype=torch.long)

    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    return data

def smiles_to_pyg_graph_2d(smiles):
    """
    Converts smile to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string")
    
    # atoms
    AllChem.ComputeGasteigerCharges(molecule)

    # Add hydrogens
    mol = Chem.AddHs(molecule)

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = \
        [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
        [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())] + \
        [allowable_features['possible_hybridization_list'].index(atom.GetHybridization())]

        atom_features_list.append(atom_feature)
        #print(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features['possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data