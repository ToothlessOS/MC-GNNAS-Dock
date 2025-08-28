#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import rdkit, rdkit.Chem, rdkit.Chem.rdDepictor, rdkit.Chem.Draw
import networkx as nx
import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
import torch
from rdkit.Chem import AllChem
# import dmol
from graphein.protein.config import ProteinGraphConfig
import graphein.molecule as gm


# In[ ]:


from functools import partial
config = gm.MoleculeGraphConfig(
    node_metadata_functions=[
        gm.atom_type_one_hot,
        gm.atomic_mass,
        gm.degree,
        gm.total_degree,
        gm.total_valence,
        gm.explicit_valence,
        gm.implicit_valence,
        gm.num_explicit_h,
        gm.num_implicit_h,
        gm.total_num_h,
        gm.num_radical_electrons,
        gm.formal_charge,
        gm.is_aromatic,
        gm.is_isotope,
        gm.is_ring,
        partial(gm.is_ring_size, ring_size=5),
        partial(gm.is_ring_size, ring_size=7)
    ]
)
config.dict()


# In[ ]:


import os
# Change here for different dataset

path = '../dataset/test'
proteins = pd.Series(os.listdir(path))

# Function to construct graphs
def construct_ligand_graph(protein_name, path):
    ligand_path = os.path.join(path, protein_name, f'{protein_name}_ligand.sdf')
    return gm.construct_graph(config=config, path=ligand_path)

# Construct graphs for all proteins listed in the DataFrame
graphs = proteins.apply(lambda p: construct_ligand_graph(p, path))




# In[ ]:


import torch
from torch_geometric.data import Data
import numpy as np

def convert_to_pyg(graph):
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}  # Map nodes to indices
    bond_type_mapping = {
        'SINGLE': 0,
        'DOUBLE': 1,
        'TRIPLE': 2,
        'AROMATIC': 3
    }

    # Node features
    node_features = []
    for node, node_data in graph.nodes(data=True):
        coords = np.array(node_data['coords'])
        one_hot = np.array(node_data['atom_type_one_hot'])
        mass = np.array(node_data['mass'])
        degree = np.array(node_data['degree'])
        total_degree = np.array(node_data['total_degree'])
        total_valence = np.array(node_data['total_valence'])
        explicit_valence = np.array(node_data['explicit_valence'])
        num_explicit_h = np.array(node_data['num_explicit_h'])
        num_implicit_h = np.array(node_data['num_implicit_h'])
        total_num_h = np.array(node_data['total_num_h'])
        num_radical_electrons = np.array(node_data['num_radical_electrons'])
        is_aromatic = np.array([node_data['is_aromatic']], dtype=np.float32)
        is_ring = np.array([node_data['is_ring']], dtype=np.float32)
        feature = np.concatenate([
            coords,
            one_hot,
            [mass],
            [degree],
            [total_degree],
            [total_valence],
            [explicit_valence],
            [num_explicit_h],
            [num_implicit_h],
            [total_num_h],
            [num_radical_electrons],
            is_aromatic,
            is_ring
        ])
        node_features.append(feature)
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Edge indices and features
    edge_indices = []
    edge_attrs = []
    for u, v, edge_data in graph.edges(data=True):
        if 'bond' in edge_data:
        # Ensure that u and v are mapped to integers
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            edge_indices.append([u_idx, v_idx])
            edge_indices.append([v_idx, u_idx])

            # Ensure edge attributes are the correct format, e.g., a single float or int per edge
            # This assumes interaction_type is a single value; adjust if it's more complex
            bond = edge_data.get('bond')
            bond_type = bond_type_mapping.get(bond.GetBondType().name, -1)
            interaction_type = np.array([bond_type], dtype=np.float32)
            edge_attrs.append(interaction_type)
            edge_attrs.append(interaction_type)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    # Create a PyTorch Geometric Data object
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
# Convert all graphs
pyg_graphs = [convert_to_pyg(g) for g in graphs]
# pyg_graphs = convert_to_pyg(graph)
# Assuming pyg_graphs is a list containing your graph data objects
proteins = proteins.to_list()
out_dir = '../ligand_g'
os.makedirs(out_dir, exist_ok=True)
for idx, pyg_graph in enumerate(pyg_graphs):
    protein_name = proteins[idx]
    file_name = f"{out_dir}/pyg_graph_{protein_name}.pt"
    torch.save(pyg_graph, file_name)
    print(f'{protein_name} saved')

