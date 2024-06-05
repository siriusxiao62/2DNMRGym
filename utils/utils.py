import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
import os 
import pickle
import cairosvg
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


from rdkit.Chem import rdMolTransforms
from rdkit.Chem import PyMol

# Function to visualize the 3D structure of a molecule from a SMILES string
def visualize_smiles_3d(smiles):
    # Create a molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    
    # Set up PyMol visualization
    viewer = PyMol.MolViewer()
    viewer.ShowMol(mol, name="Molecule")
    
    return viewer




def draw_2d_graph_cindex(smile, filename):
    mol = Chem.MolFromSmiles(smile)
    # mol_with_h = Chem.AddHs(mol)  # Ensure hydrogens are added if they are in the prediction
    AllChem.Compute2DCoords(mol)

    # Create a drawer with desired size
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)

    opts = drawer.drawOptions()
    # Show atom indices for carbon atoms
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            opts.atomLabels[atom.GetIdx()] = str(atom.GetIdx())

    # Draw the molecule with highlighted carbons
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()

    # svg_content = svg.replace('svg', '')
    svg_content = svg.replace('xmlns:svg="http://www.w3.org/2000/svg" ', '')
    # display(SVG(svg.replace('svg:', '')))

    # Save SVG content to a PNG file using cairosvg
    cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=filename)


def plot_molgraph_spectrum(file_path, visual_path, filename, h_pred, c_pred, hnmr, cnmr, ch_idx, marker_size = 20, title_size = 15, label_size = 15, tick_size = 15):
    # Load the molecular graph
    img = mpimg.imread(file_path)

    # Create a figure and specify the grid
    fig = plt.figure(figsize=(9, 5))  # You can adjust the overall size of the figure
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])  # Width ratios between the two plots

    # First subplot
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img)
    ax1.axis('off')

    # Second subplot
    ax2 = fig.add_subplot(gs[1])
    for i in range(h_pred.shape[1]):
        ax2.scatter(h_pred[:, i] * 10, c_pred[:] * 200,  c='orange', s=marker_size, marker='o', label='Prediction' if i == 0 else "")
        # Add C index as label to the prediction point
        for j in range(h_pred.shape[0]):
            if i == 0:  # If you only want to label the first set of points
                ax2.text(h_pred[j, i] * 10 + 0.08, c_pred[j] * 200 - 1, f'{ch_idx[j]}', fontsize=9)

    for i in range(h_pred.shape[1]):
        ax2.scatter(hnmr[:, i] * 10, cnmr[:] * 200, c='blue', s=marker_size, marker='^', label='Observed' if i == 0 else "")
        for j in range(h_pred.shape[0]):
            if i == 0 or hnmr[j, 0] != hnmr[j, 1]:  # If you only want to label the first set of points
                ax2.text(hnmr[j, i] * 10, cnmr[j] * 200, f'{ch_idx[j]}', fontsize=9)

    ax2.set_xlabel('H-shifts', fontsize=label_size)
    ax2.set_ylabel('C-shifts', fontsize=label_size)
    ax2.tick_params(axis='both', which='major', labelsize=tick_size)
    ax2.tick_params(axis='y', which='both', labelleft=False, labelright=True)
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.legend(fontsize=title_size)

    plt.tight_layout()
    # plt.show()
    plt.close()

    fig.savefig(os.path.join(visual_path, '%s_molecule_val.png'%filename))

