import inspect
import torch
import numpy as np
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
# TODO: Implement NDCG Loss
#from pytorchltr.loss import LambdaNDCGLoss1

from model_blocks.models_protein import GCN_GAT_GINE, GCN_GAT, GCN_GINE, GAT_GINE, GCN, GAT, GINE, GCN_GAT_GINE_FIXED_OUT_DIM
from model_blocks.models_ligand import GCN_L, GAT_L, GINE_L, SAGE_L, GAT_L_NO_FIXED_OURPUT_DIM

from utils import ndcg_score

from model.ndcg_loss import LambdaNDCGLoss2
from model.additive_loss import PairwiseLogisticLoss

# Map from model name to class
model_map = {
    'GCN_GAT_GINE': GCN_GAT_GINE,
    'GCN_GAT_GINE_fixed_out_dim': GCN_GAT_GINE_FIXED_OUT_DIM,
    'GCN_GAT': GCN_GAT,
    'GCN_GINE': GCN_GINE,
    'GAT_GINE': GAT_GINE,
    'GCN': GCN,
    'GAT': GAT,
    'GINE': GINE,
    'GCN_L': GCN_L,
    'GAT_L': GAT_L,
    'GAT_L_fixed_out_dim': GAT_L_NO_FIXED_OURPUT_DIM,
    'GINE_L': GINE_L,
    'SAGE_L': SAGE_L,
}

import pytorch_lightning as pl

#TODO: Implement seperately for binary and point-based models
class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        pl.seed_everything(kargs.get('seed', 42), workers=True)
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.ndcg_loss=LambdaNDCGLoss2(sigma=kargs.get('sigma', 1))
        self.logistic_loss = PairwiseLogisticLoss()

    def forward(self, protein, ligand):
        return self.model(protein, ligand).view(-1)

    def training_step(self, batch, batch_idx):
        if self.criteria == 'rank':
            ligand_data, protein_data = batch
            out = self(ligand_data, protein_data)
            # Reshape out and labels to match the expected dimensions
            scores = out.view(-1, self.hparams.num_classes)
            relevances = ligand_data.y.view(-1, self.hparams.num_classes)   
            # Create query_lengths tensor
            query_lengths = torch.full(
                size=(scores.shape[0],),
                fill_value=self.hparams.num_classes,
                dtype=torch.long,
                device=out.device)
            
            # Initialize loss with base loss function
            loss = self.loss_function(out, ligand_data.y.float())
            
            # Add NDCG loss if enabled
            use_ndcg = getattr(self.hparams, 'use_ndcg_loss', True)  # Default to True for backward compatibility
            ndcg_weight = getattr(self.hparams, 'ndcg_loss_weight', 1.0)
            
            if use_ndcg:
                ndcg_loss = self.ndcg_loss(
                    scores=scores,
                    relevance=relevances,
                    n=query_lengths 
                )
                # For the cases where batch size > 1
                if ndcg_loss.numel() > 1:
                    ndcg_loss = ndcg_loss.mean()
                loss += ndcg_weight * ndcg_loss
            
            # Add logistic loss if enabled
            use_logistic = getattr(self.hparams, 'use_logistic_loss', False)  # Default to False
            logistic_weight = getattr(self.hparams, 'logistic_loss_weight', 0.01)
            
            if use_logistic:
                logistic_loss = self.logistic_loss(
                    scores=scores,
                    relevance=relevances,
                    n=query_lengths 
                )
                if logistic_loss.numel() > 1:
                    logistic_loss = logistic_loss.mean()
                loss += logistic_weight * logistic_loss
            
            # Log the loss
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size, sync_dist=True)
            return loss
        elif self.criteria == 'binary' or self.criteria == 'point':
            ligand_data, protein_data = batch
            out = self(ligand_data, protein_data)
            labels = ligand_data.y.float()
            loss = self.loss_function(out, labels)
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size, sync_dist=True)
            return loss

    def validation_step(self, batch, batch_idx):
        ligand_data, protein_data = batch
        out = self(ligand_data, protein_data)
        labels = ligand_data.y.float()
        # Criteria: Binary (predict pass-or-fail)
        if self.criteria == 'binary':
            loss = self.loss_function(out, labels)
            # Compute AS accuracy
            probs = torch.sigmoid(out)
            # For binary classification, threshold at 0.5
            preds = (probs > 0.5).float()
            correct = (preds == labels).float()
            as_acc = correct.mean()
        # Criteria: Point-based (predict absolute performance)
        elif self.criteria == 'point':
            # out: (batch_size, num_algorithms)
            # labels: (batch_size, num_algorithms)
            loss = self.loss_function(out, labels)
            # Implement nDCG as a better algorithm selection metric
            batch_ndcg = []
            out_reshaped = out.view(-1, self.hparams.num_classes)
            labels_reshaped = labels.view(-1, self.hparams.num_classes)
            for i in range(out_reshaped.size(0)):
                ndcg = ndcg_score(labels_reshaped[i], out_reshaped[i], self.hparams.ndcg_k)
                batch_ndcg.append(ndcg)
            as_acc = torch.tensor(batch_ndcg, device=loss.device).float().mean()
        # TODO: Criteria: Rank (predict relative performance)
        elif self.criteria == 'rank':
            scores = out.view(-1, self.hparams.num_classes)  # [batch_size, max_docs]
            relevances = labels.view(-1, self.hparams.num_classes)
            query_lengths = torch.full(
                size=(scores.shape[0],),
                fill_value=self.hparams.num_classes,
                dtype=torch.long,
                device=out.device)
            
            # Initialize loss with base loss function
            loss = self.loss_function(out, ligand_data.y.float())
            
            # Add NDCG loss if enabled
            use_ndcg = getattr(self.hparams, 'use_ndcg_loss', True)
            ndcg_weight = getattr(self.hparams, 'ndcg_loss_weight', 1.0)
            
            if use_ndcg:
                ndcg_loss = self.ndcg_loss(
                    scores=scores,
                    relevance=relevances,
                    n=query_lengths 
                )
                if ndcg_loss.numel() > 1:
                    ndcg_loss = ndcg_loss.mean()
                loss += ndcg_weight * ndcg_loss
            
            # Add logistic loss if enabled
            use_logistic = getattr(self.hparams, 'use_logistic_loss', False)
            logistic_weight = getattr(self.hparams, 'logistic_loss_weight', 0.01)
            
            if use_logistic:
                logistic_loss = self.logistic_loss(
                    scores=scores,
                    relevance=relevances,
                    n=query_lengths 
                )
                if logistic_loss.numel() > 1:
                    logistic_loss = logistic_loss.mean()
                loss += logistic_weight * logistic_loss
            
            # AS accuracy
            batch_ndcg = []
            for i in range(scores.shape[0]):
                ndcg = ndcg_score(relevances[i], scores[i], self.hparams.ndcg_k)
                batch_ndcg.append(ndcg)
            as_acc = torch.tensor(batch_ndcg, device=loss.device).float().mean()
        else:
            as_acc = torch.tensor(0.0, device=loss.device)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_as_acc', as_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log learning rate
        opt = self.optimizers() if hasattr(self, "optimizers") else None
        if opt is not None:
            lr = opt.param_groups[0]['lr']
            self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_as_acc': as_acc, 'lr': lr}

    def test_step(self, batch, batch_idx):
        ligand_data, protein_data = batch
        out = self(ligand_data, protein_data)
        labels = ligand_data.y.float()
        # Criteria: Binary (predict pass-or-fail)
        if self.criteria == 'binary':
            loss = self.loss_function(out, labels)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            correct = (preds == labels).float()
            as_acc = correct.mean()
        # Criteria: Point-based (predict absolute performance)
        elif self.criteria == 'point':
            # out: (batch_size, num_algorithms)
            # labels: (batch_size, num_algorithms)
            loss = self.loss_function(out, labels)
            # Implement nDCG as a better algorithm selection metric
            batch_ndcg = []
            out_reshaped = out.view(-1, self.hparams.num_classes)
            labels_reshaped = labels.view(-1, self.hparams.num_classes)
            for i in range(out_reshaped.size(0)):
                ndcg = ndcg_score(labels_reshaped[i], out_reshaped[i], self.hparams.ndcg_k)
                batch_ndcg.append(ndcg)
            as_acc = torch.tensor(batch_ndcg, device=loss.device).float().mean()
        # TODO: Criteria: Rank (predict relative performance)
        elif self.criteria == 'rank':
            scores = out.view(-1, self.hparams.num_classes)  # [batch_size, max_docs]
            relevances = labels.view(-1, self.hparams.num_classes)
            query_lengths = torch.full(
                size=(scores.shape[0],),
                fill_value=self.hparams.num_classes,
                dtype=torch.long,
                device=out.device)

              # Initialize loss with base loss function
            loss = self.loss_function(out, ligand_data.y.float())
            
            # Add NDCG loss if enabled
            use_ndcg = getattr(self.hparams, 'use_ndcg_loss', True)
            ndcg_weight = getattr(self.hparams, 'ndcg_loss_weight', 1.0)
            
            if use_ndcg:
                ndcg_loss = self.ndcg_loss(
                    scores=scores,
                    relevance=relevances,
                    n=query_lengths 
                )
                if ndcg_loss.numel() > 1:
                    ndcg_loss = ndcg_loss.mean()
                loss += ndcg_weight * ndcg_loss
            
            # Add logistic loss if enabled
            use_logistic = getattr(self.hparams, 'use_logistic_loss', False)
            logistic_weight = getattr(self.hparams, 'logistic_loss_weight', 0.01)
            
            if use_logistic:
                logistic_loss = self.logistic_loss(
                    scores=scores,
                    relevance=relevances,
                    n=query_lengths 
                )
                if logistic_loss.numel() > 1:
                    logistic_loss = logistic_loss.mean()
                loss += logistic_weight * logistic_loss
            
            # AS accuracy
            batch_ndcg = []
            for i in range(scores.size(0)):
                ndcg = ndcg_score(relevances[i], scores[i], self.hparams.ndcg_k)
                batch_ndcg.append(ndcg)
            as_acc = torch.tensor(batch_ndcg, device=loss.device).float().mean()
        else:
            as_acc = torch.tensor(0.0, device=loss.device)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_as_acc', as_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Create output dictionary
        output_dict = {
            'test_loss': loss, 
            'test_as_acc': as_acc, 
            'out': out.detach().cpu().numpy(), 
            'labels': labels.detach().cpu().numpy(),
            'names': ligand_data.names if hasattr(ligand_data, 'names') else None,
            # TODO: Compatibility with binary models (old model) 
        }
        
        # Explicitly append to the test_step_outputs list we initialize in on_test_epoch_start
        if hasattr(self, "test_step_outputs"):
            self.test_step_outputs.append(output_dict)
    
        return output_dict
    
    def on_test_epoch_end(self, outputs=None):
        # Access the outputs through the dataloader_outputs attribute
        if not hasattr(self, "test_step_outputs") or not self.test_step_outputs:
            print("No outputs collected from test_step")
            self.test_results = {"out": None, "labels": None}
            return
        
        # Process the collected outputs
        all_out = np.concatenate([o["out"] for o in self.test_step_outputs])
        all_labels = np.concatenate([o["labels"] for o in self.test_step_outputs])
        all_names = np.concatenate([o["names"] for o in self.test_step_outputs])
        
        # Save for later use
        self.test_results = {"out": all_out, "labels": all_labels, 'names': all_names}
        
        # Clear the outputs to free memory
        self.test_step_outputs = []

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
    def configure_optimizers(self):
        # Configure a lr discount for the GNN layers to ensure stable training
        gnn_params = []
        other_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                # Check if parameter belongs to GNN layers
                if any(gnn_layer in name for gnn_layer in ['gcn', 'gat', 'gin', 'sage', 'conv']):
                    gnn_params.append(param)
                else:
                    other_params.append(param)

        param_groups = []
        if gnn_params:
            param_groups.append({
                'params': gnn_params,
                'lr': self.hparams.lr * 0.5,
                'name': 'gnn_layers'
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.hparams.lr * 1,
                'name': 'other_layers'
            })

        # Adam optimizer, lr from hparams, no weight decay by default
        optimizer = torch.optim.Adam(
            param_groups
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.95, patience=4, min_lr=1e-6
        )
        # ALTN: Custom scheduler with warm-up and cosine annealing
        warmup_epochs = getattr(self.hparams, 'warmup_epochs', 10)
        max_epochs = getattr(self.hparams, 'max_epochs', 100)
        cycle_length = getattr(self.hparams, 'cycle_length', 20)
        
        def combined_lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warm-up phase
                return epoch / warmup_epochs
            else:
                # After warm-up: combine decay with cyclic behavior
                progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
                
                # Exponential decay component
                decay_factor = 0.99 ** ((epoch - warmup_epochs) // 10)
                
                # Cyclic component (triangular wave)
                cycle_progress = ((epoch - warmup_epochs) % cycle_length) / cycle_length
                cyclic_factor = 1.0 + 0.5 * np.sin(2 * np.pi * cycle_progress)
                
                return decay_factor * cyclic_factor
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, combined_lr_lambda)


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'bce_with_logits':
            self.loss_function = F.binary_cross_entropy_with_logits
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'ce':
            self.loss_function = F.cross_entropy
        elif loss == 'mse':
            self.loss_function = F.mse_loss
        elif self.hparams.loss == 'ndcg':
            self.loss_function = NDCGLoss(sigma=self.hparams.sigma)
        else:
            raise ValueError(f"Invalid loss type: {loss}")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        # Get model identifiers from hparams or defaults
        num_classes = getattr(self.hparams, 'num_classes')
        dropout_rate = getattr(self.hparams, 'dropout_rate', 0.3)
        
        # AS criteria (binary, point-based, rank, etc.)
        self.criteria = getattr(self.hparams, 'model_name')

        # ligand model
        model1_name = getattr(self.hparams, 'model1', 'GAT_L')
        model1_features = getattr(self.hparams, 'model1_features', 25)
        # protein model
        model2_name = getattr(self.hparams, 'model2', 'GCN_GAT_GINE')
        model2_features = getattr(self.hparams, 'model2_features', 1280)
        
        # Instantiate sub-models
        model1 = model_map[model1_name](model1_features, num_classes)  # adjust input features as needed
        model2 = model_map[model2_name](model2_features, num_classes)  # adjust input features as needed
        # Prepare arguments for the main model
        args1 = {
            'model1': model1,
            'model2': model2,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate
        }
        args1.update(other_args)
        return Model(**args1)
    
# TODO: Fix graident flow issues! (Class currently unused)
class NDCGLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, scores, relevances, query_lengths):
        
        batch_size, max_docs = scores.shape
        loss = 0.0
        
        for i in range(batch_size):
            n_docs = query_lengths[i].item()
            s = scores[i, :n_docs] 
            y = relevances[i, :n_docs]
            
        
            diff = s.view(-1,1) - s.view(1,-1)  
            pairwise_gain = torch.pow(2.0, y) - 1.0
            delta = torch.abs(pairwise_gain.view(-1,1) - pairwise_gain.view(1,-1))
            
            
            discount = 1.0 / torch.log2(torch.arange(n_docs, device=s.device) + 2.0)
            ideal_dcg = torch.sum(torch.sort(pairwise_gain, descending=True)[0] * discount)
            
            
            prob = 1.0 / (1 + torch.exp(-self.sigma * diff))
            loss += -torch.sum(delta * torch.log2(prob)) / (ideal_dcg + 1e-8)
            
        return loss / batch_size

