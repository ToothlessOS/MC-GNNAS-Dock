import random
import pandas as pd
import os

import pytorch_lightning as pl
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader, DataListLoader
from .dataset import CombinedDataset, CombinedInferenceDataset, prepare_data_binary, prepare_data_point
from sklearn.model_selection import train_test_split, KFold

class DInterface(pl.LightningDataModule):
    def __init__(self, num_workers=8, dataset='', **kwargs):
        super().__init__()
        pl.seed_everything(kwargs.get('seed', 42), workers=True)
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.model_name = kwargs.get('model_name')
        self.batch_size = kwargs.get('batch_size', 8)
        self.drop_columns = kwargs.get('drop_columns', [])
        self.test_size = kwargs.get('test_size', 0.1)
        self.seed = kwargs.get('seed', 42)
        self.k_folds = kwargs.get('k_folds', None)  # None means no K-Fold, just train/val/test split
        self.current_fold = kwargs.get('fold_num', 0)  # Default to first fold

        # Prepare dataset differently based on AS criteria
        if self.model_name == 'binary':
            self.dataset, self.num_classes = prepare_data_binary(self.drop_columns)
        elif self.model_name == 'point':
            self.dataset, self.num_classes = prepare_data_point(self.drop_columns)
        elif self.model_name == 'rank':
            self.dataset, self.num_classes = prepare_data_point(self.drop_columns)
            # To do
            #TODO: Implement rank dataset preparation
        self.load_data_module()

    def setup(self, stage=None):
        if self.k_folds is not None:
            print("K-Fold enabled")            
            self.kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            self.folds = list(self.kf.split(self.dataset))
            train_idx, val_idx = self.folds[self.current_fold]
            
            self.trainset = self.dataset.iloc[train_idx]
            self.valset = self.dataset.iloc[val_idx]
            self.testset = self.valset # For simplicity, using valset as testset in K-Fold
        else:
            print("K-Fold disabled")
            train_val, self.testset = train_test_split(self.dataset, test_size=self.test_size, random_state=self.seed)
            self.trainset, self.valset = train_test_split(train_val, test_size=self.test_size, random_state=self.seed)

    def train_dataloader(self):
        # Combining the protein and ligand graphs into the dataset
        trainset_combined = CombinedDataset(self.trainset)
        return DataLoader(trainset_combined, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        valset_combined = CombinedDataset(self.valset)
        return DataLoader(valset_combined, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        testset_combined = CombinedDataset(self.testset)
        return DataLoader(testset_combined, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    

    # These are from the template, but not used in our implementation.
    def load_data_module(self):
        pass

    def instancialize(self, **other_args):
        pass

class DInferenceInterface(pl.LightningDataModule):
    def __init__(self, protein_graph_dir, ligand_graph_dir, num_workers=8, **kwargs):
        super().__init__()
        self.protein_graph_dir = protein_graph_dir
        self.ligand_graph_dir = ligand_graph_dir
        self.num_workers = num_workers
        self.batch_size = kwargs.get('batch_size', 8)
        self.kwargs = kwargs
        
        # Validate directories exist
        if not os.path.exists(protein_graph_dir):
            raise FileNotFoundError(f"Protein graph directory not found: {protein_graph_dir}")
        if not os.path.exists(ligand_graph_dir):
            raise FileNotFoundError(f"Ligand graph directory not found: {ligand_graph_dir}")
        
        # Get list of available graphs and find common protein-ligand pairs
        self._find_common_pairs()

    def _find_common_pairs(self):
        """Find protein-ligand pairs that have both protein and ligand graphs."""
        # Get ligand graph files
        ligand_files = [f for f in os.listdir(self.ligand_graph_dir) if f.startswith('pyg_graph_') and f.endswith('.pt')]
        ligand_names = [f.replace('pyg_graph_', '').replace('.pt', '') for f in ligand_files]
        
        # Get protein graph files  
        protein_files = [f for f in os.listdir(self.protein_graph_dir) if f.startswith('pyg_graph_') and f.endswith('.pt')]
        protein_names = [f.replace('pyg_graph_', '').replace('.pt', '') for f in protein_files]
        
        # Find common pairs
        self.names_list = list(set(ligand_names) & set(protein_names))
        
        if len(self.names_list) == 0:
            raise ValueError("No common protein-ligand pairs found between the two directories.")
        
        print(f"Found {len(self.names_list)} protein-ligand pairs for inference.")

    def setup(self, stage=None):
        # For inference, we use all available data
        pass

    def predict_dataloader(self):
        """DataLoader for inference/prediction."""
        inference_dataset = CombinedInferenceDataset(
            self.names_list, 
            self.ligand_graph_dir, 
            self.protein_graph_dir
        )
        return DataLoader(
            inference_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            persistent_workers=True if self.num_workers > 0 else False
        )

    def get_names_list(self):
        """Return the list of protein-ligand pair names."""
        return self.names_list