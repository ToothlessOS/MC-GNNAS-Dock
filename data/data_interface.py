import random
import pandas as pd

import pytorch_lightning as pl
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader, DataListLoader
from .dataset import CombinedDataset, prepare_data_binary, prepare_data_point
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
            train_val, self.testset = train_test_split(self.dataset, test_size=self.test_size, random_state=self.seed)
            
            self.kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            self.folds = list(self.kf.split(train_val))
            train_idx, val_idx = self.folds[self.current_fold]
            
            self.trainset = train_val.iloc[train_idx]
            self.valset = train_val.iloc[val_idx]
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