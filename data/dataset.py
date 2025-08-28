import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import pandas as pd
import math

# Dataset classes
# ligand graph
class SMILESDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
                    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        ligand_name = self.dataframe.iloc[idx]['ligand']
        protein_name = self.dataframe.iloc[idx]['protein']
        names = f'{protein_name}_{ligand_name}'
        nodes, edge_index, edge_attr = torch.load(f'dataset/ligand_g_mod/pyg_graph_{names}.pt') # x, edge_index, edge_attr will make tuples in the pt. use [1] to load
        target = self.dataframe.iloc[idx]['target'].astype(float)
        graph_data = Data(x=nodes[1], edge_index=edge_index[1], edge_attr=edge_attr[1], y=torch.tensor(target, dtype=torch.float32))
        graph_data.names = names
        return graph_data
    
# protein graph
class GraphDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
                    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        protein_name = self.dataframe.iloc[idx]['protein']
        ligand_name = self.dataframe.iloc[idx]['ligand']
        names = f'{protein_name}_{ligand_name}'
        nodes, edge_index, edge_attr = torch.load(f'dataset/protein_g_mod/pyg_graph_{protein_name}.pt') # x, edge_index, edge_attr will make tuples in the pt. use [1] to load
        target = self.dataframe.iloc[idx]['target'].astype(float)
        graph_data = Data(x=nodes[1], edge_index=edge_index[1], edge_attr=edge_attr[1][:, 1:2], y=torch.tensor(target, dtype=torch.float32))# dtype=torch.float32
        graph_data.names = names
        return graph_data
    
# Create Combined Dataset for use in stacked model
class CombinedDataset(Dataset):
    def __init__(self, dataframe):
        self.smiles_dataset = SMILESDataset(dataframe)
        self.graph_dataset = GraphDataset(dataframe)

    def __len__(self):
        return len(self.smiles_dataset)

    def __getitem__(self, idx):
        # Get data from both datasets
        smiles_data = self.smiles_dataset[idx]
        graph_data = self.graph_dataset[idx]
        
        # Since both dataframes are the same, ligand names should match
        assert smiles_data.names == graph_data.names, "names do not match."
        
        return smiles_data, graph_data

# Prepare dataset for the binary (go or no-go) algorithm selection task  
def prepare_data_binary(drop_columns:list = []):
    # Load the 'pose' and 'rmsd' datasets
    pose = pd.read_csv('dataset/moad_pose_check(no_rmsd).csv').iloc[:, 1:]
    rmsd = pd.read_csv('dataset/rmsd_train.csv')

    # Merge into a single DataFrame
    protein = [i.split('_')[0] for i in rmsd.name]
    ligand = ['_'.join(i.split('_')[1:]) for i in rmsd.name]
    rmsd.insert(0, 'protein', protein)
    rmsd.insert(1, 'ligand', ligand)
    rmsd = rmsd.drop(columns=['name'])
    rmsd.columns=['protein', 'ligand', 'surf', 'uni', 'gnina', 'smina', 'qvina', 'diff', 'diffL', 'karma']
    pose.columns=['protein', 'ligand', 'diffL', 'diff', 'gnina', 'karma', 'surf', 'uni']
    pose['smina'] = True
    pose['qvina'] = True
    pose = pose[rmsd.columns] #important: align columns
    
    rmsd.iloc[:, 2:] = rmsd.iloc[:, 2:] <= 2
    final_df = pose.iloc[:, 2:] & rmsd.iloc[:, 2:] # True only if both 'pose' and 'rmsd' are True (both criteria satisfied)
    final_df.insert(0, 'protein', protein)
    final_df.insert(1, 'ligand', ligand)

    # Drop the unused columns(algorithms)
    if len(drop_columns) > 0:
        final_df = final_df.drop(columns=drop_columns)

    target = [final_df.iloc[i,2:].values for i in range(len(final_df))]
    final_df['target'] = target
    num_classes = len(final_df.columns[2:-1])
    return final_df, num_classes

# TODO: prepare dataset for the point-based algorithm selection task
def prepare_data_point(drop_columns:list = []):
    # Load the 'pose' and 'rmsd' datasets
    pose = pd.read_csv('dataset/moad_pose_check(no_rmsd).csv').iloc[:, 1:]
    pb_ratios = pd.read_csv('dataset/moad_pose_ratio(no_rmsd).csv').iloc[:, 1:]
    rmsd = pd.read_csv('dataset/rmsd_train.csv')

    # Merge into a single DataFrame
    protein = [i.split('_')[0] for i in rmsd.name]
    ligand = ['_'.join(i.split('_')[1:]) for i in rmsd.name]
    rmsd.insert(0, 'protein', protein)
    rmsd.insert(1, 'ligand', ligand)
    rmsd = rmsd.drop(columns=['name'])
    rmsd.columns=['protein', 'ligand', 'surf', 'uni', 'gnina', 'smina', 'qvina', 'diff', 'diffL', 'karma']
    pose.columns=['protein', 'ligand', 'diffL', 'diff', 'gnina', 'karma', 'surf', 'uni']
    pb_ratios.columns=['protein', 'ligand', 'diffL', 'diff', 'gnina', 'karma', 'surf', 'uni', 'smina', 'qvina']
    pose['smina'] = True
    pose['qvina'] = True
    pose = pose[rmsd.columns] #important: align columns
    pb_ratios = pb_ratios[rmsd.columns]
    pb_ratios['smina'] = 1
    pb_ratios['qvina'] = 1

       # Scoring function setup (new)
    alpha = 0.5 # Weight for the RMSD score

    def pose_score(x, lambda_pb=100):
        return math.exp(lambda_pb * (x-1)) if x >= 0 else 0
      
    # RMSD scores
    def rmsd_score(x, lambda_rmsd=1):
        if x > 5 or x < 0:
            return 0
        score = (11 - math.exp(lambda_rmsd*x))/10
        if score >= 0:
            return score
        else:
            return 0
        
    # Score computation
    pose_scores = pose.iloc[:, 2:].astype(int)
    rmsd_scores = rmsd.iloc[:, 2:].applymap(rmsd_score)
    final_df = (1-alpha)*pose_scores + alpha*rmsd_scores
    final_df.insert(0, 'protein', protein)
    final_df.insert(1, 'ligand', ligand)

    # Drop the unused columns(algorithms)
    if len(drop_columns) > 0:
        final_df = final_df.drop(columns=drop_columns)

    target = [final_df.iloc[i,2:].values for i in range(len(final_df))]
    final_df['target'] = target
    num_classes = len(final_df.columns[2:-1])

    return final_df, num_classes

# Directly return the RMSD scores; Currently used for evaluation only!
def prepare_data_rmsd(drop_columns:list = []):
    # Load the 'pose' and 'rmsd' datasets
    pose = pd.read_csv('dataset/moad_pose_check(no_rmsd).csv').iloc[:, 1:]
    rmsd = pd.read_csv('dataset/rmsd_train.csv')

    # Merge into a single DataFrame
    protein = [i.split('_')[0] for i in rmsd.name]
    ligand = ['_'.join(i.split('_')[1:]) for i in rmsd.name]
    rmsd.insert(0, 'protein', protein)
    rmsd.insert(1, 'ligand', ligand)
    rmsd = rmsd.drop(columns=['name'])
    rmsd.columns=['protein', 'ligand', 'surf', 'uni', 'gnina', 'smina', 'qvina', 'diff', 'diffL', 'karma']
    pose.columns=['protein', 'ligand', 'diffL', 'diff', 'gnina', 'karma', 'surf', 'uni']
    pose['smina'] = True
    pose['qvina'] = True
    pose = pose[rmsd.columns] #important: align columns

    final_df = rmsd

    # Drop the unused columns(algorithms)
    if len(drop_columns) > 0:
        final_df = final_df.drop(columns=drop_columns)

    target = [final_df.iloc[i,2:].values for i in range(len(final_df))]
    final_df['target'] = target
    num_classes = len(final_df.columns[2:-1])

    return final_df, num_classes

# Directly return the PoseBuster pass (or not); Currently used for evaluation only!
def prepare_data_pose(drop_columns:list = []):
    # Load the 'pose' and 'rmsd' datasets
    pb_ratios = pd.read_csv('dataset/moad_pose_ratio(no_rmsd).csv').iloc[:, 1:]
    rmsd = pd.read_csv('dataset/rmsd_train.csv')
    pose = pd.read_csv('dataset/moad_pose_check(no_rmsd).csv').iloc[:, 1:]

    # Merge into a single DataFrame
    protein = [i.split('_')[0] for i in rmsd.name]
    ligand = ['_'.join(i.split('_')[1:]) for i in rmsd.name]
    rmsd.insert(0, 'protein', protein)
    rmsd.insert(1, 'ligand', ligand)
    rmsd = rmsd.drop(columns=['name'])
    rmsd.columns=['protein', 'ligand', 'surf', 'uni', 'gnina', 'smina', 'qvina', 'diff', 'diffL', 'karma']
    pb_ratios.columns=['protein', 'ligand', 'diffL', 'diff', 'gnina', 'karma', 'surf', 'uni', 'smina', 'qvina']
    pose.columns=['protein', 'ligand', 'diffL', 'diff', 'gnina', 'karma', 'surf', 'uni']
    pb_ratios = pb_ratios[rmsd.columns] #important: align columns
    pose['smina'] = True
    pose['qvina'] = True
    pose = pose[rmsd.columns] #important: align columns

    final_df = pose.iloc[:, 2:].astype(int) # True if PoseBuster passes, False otherwise
    final_df.insert(0, 'protein', protein)
    final_df.insert(1, 'ligand', ligand)

    # Drop the unused columns(algorithms)
    if len(drop_columns) > 0:
        final_df = final_df.drop(columns=drop_columns)

    target = [final_df.iloc[i,2:].values for i in range(len(final_df))]
    final_df['target'] = target
    num_classes = len(final_df.columns[2:-1])

    return final_df, num_classes