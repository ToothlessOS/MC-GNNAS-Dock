# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model import MInterface
from data import DInterface
from utils import load_model_path_by_args, plot_rmsd_metrics

from data.dataset import prepare_data_binary, prepare_data_point, prepare_data_pose, prepare_data_rmsd
from sklearn.model_selection import train_test_split, KFold
from scipy import stats

from utils import ndcg_score

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required for CUDA >=10.2
os.environ["PYTHONHASHSEED"] = "0"                 # For Python/NumPy reproducibility

import math

def load_callbacks(args):
    callbacks = []
    
    # Remove early (comment this out if not needed)
    """
    callbacks.append(plc.EarlyStopping(
        monitor='val_as_acc',
        mode='max',
        patience=10,
        min_delta=0.001
    ))
    """

    # Options: val_loss(min), val_as_acc(max)
    callbacks.append(plc.ModelCheckpoint(
        monitor='val_as_acc',
        filename='best-{epoch:03d}-{val_loss:.4f}-{val_as_acc:.4f}',
        save_top_k=3,
        mode='max',
        save_last=True
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='low-val-loss-{epoch:03d}-{val_loss:.4f}-{val_as_acc:.4f}',
        save_top_k=3,
        mode='min',
        save_last=False
    ))

    return callbacks

# Main function for training 
def main(args):
    pl.seed_everything(args.seed) # Fix seed for reproducibility
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))

    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.ckpt_path = load_path

    callbacks = load_callbacks(args)
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)  # Uncomment and set log_dir if needed

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        # deterministic=True,  # Enables torch.backends.cudnn.deterministic
        # benchmark=False,     # Disables torch.backends.cudnn.benchmark
        # logger=logger,  # Uncomment if logger is used
        # ...other Trainer args...
    )
    trainer.fit(model, data_module)

def test(args):
    pl.seed_everything(args.seed)
    # Load model and data
    data_module = DInterface(**vars(args))
    data_module.setup()
    test_loader = data_module.test_dataloader()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MInterface.load_from_checkpoint(args.ckpt_path, map_location=device)
    model = model.eval()

    criteria = getattr(args, 'model_name')
    num_classes = getattr(args, 'num_classes')

    trainer = Trainer(accelerator="cuda", logger=False, enable_checkpointing=False, devices=[0])
    results = trainer.test(model, datamodule=data_module)
    print(results)
    
    # Load the appropriate test dataset based on the configuration
    if args.use_separate_test:
        if criteria == 'binary':
            dataset, _ = prepare_data_binary(args.drop_columns, dataset_type=args.test_dataset_type)
            pose_dataset, _ = prepare_data_pose(args.drop_columns, dataset_type=args.test_dataset_type)
            rmsd_dataset, _ = prepare_data_rmsd(args.drop_columns, dataset_type=args.test_dataset_type)
        elif criteria in ['point', 'rank']:
            dataset, _ = prepare_data_point(args.drop_columns, dataset_type=args.test_dataset_type)
            pose_dataset, _ = prepare_data_pose(args.drop_columns, dataset_type=args.test_dataset_type)
            rmsd_dataset, _ = prepare_data_rmsd(args.drop_columns, dataset_type=args.test_dataset_type)
        
        # For separate test datasets, use the entire dataset as test
        test_df = dataset
        test_df_pose = pose_dataset
        test_df_rmsd = rmsd_dataset
    else:
        # Original behavior
        if criteria == 'binary':
            dataset, _ = prepare_data_binary(args.drop_columns)
        elif criteria in ['point', 'rank']:
            dataset, _ = prepare_data_point(args.drop_columns)
        
        pose_dataset, _ = prepare_data_pose(args.drop_columns)
        rmsd_dataset, _ = prepare_data_rmsd(args.drop_columns)

        if args.k_folds is not None:
            # K-Fold cross-validation
            kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
            folds = list(kf.split(dataset))
            train_idx, val_idx = folds[args.fold_num]
            
            train_val = dataset.iloc[train_idx]
            test_df = dataset.iloc[val_idx]
            
            train_val_pose = pose_dataset.iloc[train_idx]
            test_df_pose = pose_dataset.iloc[val_idx]
            
            train_val_rmsd = rmsd_dataset.iloc[train_idx]
            test_df_rmsd = rmsd_dataset.iloc[val_idx]

        else:

            train_val, test_df = train_test_split(dataset, test_size=args.test_size, random_state=args.seed)
            train_df, val_df = train_test_split(train_val, test_size=args.test_size, random_state=args.seed)
            
            train_val_pose, test_df_pose = train_test_split(pose_dataset, test_size=args.test_size, random_state=args.seed)
            train_df_pose, val_df_pose = train_test_split(train_val_pose, test_size=args.test_size, random_state=args.seed)
            
            train_val_rmsd, test_df_rmsd = train_test_split(rmsd_dataset, test_size=args.test_size, random_state=args.seed)
            train_df_rmsd, val_df_rmsd = train_test_split(train_val_rmsd, test_size=args.test_size, random_state=args.seed)

    out = model.test_results["out"]
    labels = model.test_results["labels"]
    names = model.test_results["names"]

    test_df = test_df.reset_index(drop=True)  # Reset index to avoid KeyError
    choices = test_df.columns[2:-1]
    cap_res_arr = out.reshape(-1, num_classes)  # Ensure shape is (-1, num_classes)
    cap_df = pd.DataFrame(cap_res_arr, columns=choices) #.rank(axis=1, method='min')
    name_df = pd.DataFrame(names, columns=['Ligand'])
    cap_out = pd.concat([name_df, cap_df], axis=1)
    # Identify selected model for each protein-ligand pair
    cap_out['selected_model'] = cap_out.iloc[:,1:].idxmax(axis=1)
    print(cap_out)

    descriptive = test_df.iloc[:,:-1]
    best_method = []
    print(descriptive)

    # Create descriptive_pose and descriptive_rmsd dataframes
    descriptive_pose = test_df_pose.iloc[:,:-1]
    descriptive_rmsd = test_df_rmsd.iloc[:,:-1]
    
    # Reset indices to ensure alignment
    descriptive = descriptive.reset_index(drop=True)
    descriptive_pose = descriptive_pose.reset_index(drop=True)
    descriptive_rmsd = descriptive_rmsd.reset_index(drop=True)

    for i in range(len(descriptive)):
        temp_name = descriptive.protein[i]+'_'+descriptive.ligand[i]
        method = cap_out[cap_out.Ligand == temp_name].selected_model.values[0]
        best_method.append(method)
        
    # Apply same operations to all three dataframes
    descriptive['choose'] = best_method
    descriptive_pose['choose'] = best_method
    descriptive_rmsd['choose'] = best_method
    
    if criteria == 'binary':
        descriptive['oracle'] = descriptive.iloc[:,2:-1].any(axis=1)
        descriptive_pose['oracle'] = descriptive_pose.iloc[:,2:-1].any(axis=1)
        descriptive_rmsd['oracle'] = descriptive_rmsd.iloc[:,2:-1].any(axis=1)
        # For binary, 2nd and 3rd best don't make sense, so you may skip or set to NaN
        descriptive['2nd'] = np.nan
        descriptive['3rd'] = np.nan
        descriptive_pose['2nd'] = np.nan
        descriptive_pose['3rd'] = np.nan
        descriptive_rmsd['2nd'] = np.nan
        descriptive_rmsd['3rd'] = np.nan
    elif criteria == 'point' or criteria == 'rank':
        descriptive['oracle'] = descriptive.iloc[:,2:-1].max(axis=1)
        descriptive_pose['oracle'] = descriptive_pose.iloc[:,2:-1].max(axis=1)
        descriptive_rmsd['oracle'] = descriptive_rmsd.iloc[:,2:-1].min(axis=1)  # For RMSD, best is minimum
        
        # Ensure model_scores are numeric and compute 2nd and 3rd best for scoring data
        model_scores = descriptive.iloc[:,2:-1].apply(pd.to_numeric, errors='coerce')
        descriptive['2nd'] = model_scores.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
        descriptive['3rd'] = model_scores.apply(lambda row: row.nlargest(3).iloc[-1], axis=1)
        
        # For pose data (binary/boolean), compute 2nd and 3rd differently
        model_scores_pose = descriptive_pose.iloc[:,2:-1].apply(pd.to_numeric, errors='coerce')
        # Check if pose data is binary (0/1) or continuous
        if descriptive_pose.iloc[:,2:-1].dtypes.apply(lambda x: x in ['bool', 'int64', 'float64']).all():
            # For binary pose data, 2nd and 3rd largest make sense
            descriptive_pose['2nd'] = model_scores_pose.apply(lambda row: row.nlargest(2).iloc[-1] if len(row.dropna()) >= 2 else np.nan, axis=1)
            descriptive_pose['3rd'] = model_scores_pose.apply(lambda row: row.nlargest(3).iloc[-1] if len(row.dropna()) >= 3 else np.nan, axis=1)
        else:
            # For non-numeric pose data, set to NaN
            descriptive_pose['2nd'] = np.nan
            descriptive_pose['3rd'] = np.nan
        
        # For RMSD data, use nsmallest since lower RMSD is better
        model_scores_rmsd = descriptive_rmsd.iloc[:,2:-1].apply(pd.to_numeric, errors='coerce')
        descriptive_rmsd['2nd'] = model_scores_rmsd.apply(lambda row: row.nsmallest(2).iloc[-1] if len(row.dropna()) >= 2 else np.nan, axis=1)
        descriptive_rmsd['3rd'] = model_scores_rmsd.apply(lambda row: row.nsmallest(3).iloc[-1] if len(row.dropna()) >= 3 else np.nan, axis=1)
    
    # Apply AS calculation to all three dataframes
    values = descriptive.apply(lambda row: row[row['choose']], axis=1)
    descriptive.insert(args.num_classes+2, 'AS', values)
    
    values_pose = descriptive_pose.apply(lambda row: row[row['choose']], axis=1)
    descriptive_pose.insert(args.num_classes+2, 'AS', values_pose)
    
    values_rmsd = descriptive_rmsd.apply(lambda row: row[row['choose']], axis=1)
    descriptive_rmsd.insert(args.num_classes+2, 'AS', values_rmsd)
    
    # Generate detailed selection portfolio for AS
    portfolio_data = []
    algorithm_columns = ['surf', 'uni', 'gnina', 'smina', 'qvina', 'diff', 'diffL', 'karma']
    
    for i in range(len(descriptive)):
        protein = descriptive.iloc[i]['protein']
        ligand = descriptive.iloc[i]['ligand']
        selected_algorithm = descriptive.iloc[i]['choose']
        
        # Get AS performance metrics
        as_score = descriptive.iloc[i]['AS']
        as_pose_valid = descriptive_pose.iloc[i]['AS']
        as_rmsd = descriptive_rmsd.iloc[i]['AS']
        
        # Get oracle performance metrics
        oracle_score = descriptive.iloc[i]['oracle']
        oracle_pose_valid = descriptive_pose.iloc[i]['oracle']
        oracle_rmsd = descriptive_rmsd.iloc[i]['oracle']
        
        # Get individual algorithm performance
        algo_performances = {}
        for algo in algorithm_columns:
            if algo in descriptive.columns:
                algo_performances[f'{algo}_score'] = descriptive.iloc[i][algo]
                algo_performances[f'{algo}_pose_valid'] = descriptive_pose.iloc[i][algo]
                algo_performances[f'{algo}_rmsd'] = descriptive_rmsd.iloc[i][algo]
        
        # Get prediction scores from the model for this case
        prediction_scores = {}
        for j, algo in enumerate(algorithm_columns):
            if j < len(cap_res_arr[i]):
                prediction_scores[f'{algo}_prediction'] = cap_res_arr[i][j]
        
        portfolio_entry = {
            'protein': protein,
            'ligand': ligand,
            'selected_algorithm': selected_algorithm,
            'AS_score': as_score,
            'AS_pose_valid': as_pose_valid,
            'AS_rmsd': as_rmsd,
            'AS_rmsd_lt_1': 1 if as_rmsd < 1 else 0,
            'AS_rmsd_lt_2': 1 if as_rmsd < 2 else 0,
            'AS_rmsd_lt_1_valid': 1 if (as_rmsd < 1 and as_pose_valid) else 0,
            'AS_rmsd_lt_2_valid': 1 if (as_rmsd < 2 and as_pose_valid) else 0,
            'oracle_score': oracle_score,
            'oracle_pose_valid': oracle_pose_valid,
            'oracle_rmsd': oracle_rmsd,
            'oracle_rmsd_lt_1': 1 if oracle_rmsd < 1 else 0,
            'oracle_rmsd_lt_2': 1 if oracle_rmsd < 2 else 0,
            'oracle_rmsd_lt_1_valid': 1 if (oracle_rmsd < 1 and oracle_pose_valid) else 0,
            'oracle_rmsd_lt_2_valid': 1 if (oracle_rmsd < 2 and oracle_pose_valid) else 0,
        }
        
        # Add individual algorithm performances
        portfolio_entry.update(algo_performances)
        # Add prediction scores
        portfolio_entry.update(prediction_scores)
        
        portfolio_data.append(portfolio_entry)
    
    # Create AS portfolio DataFrame and save to CSV
    portfolio_df = pd.DataFrame(portfolio_data)
    portfolio_csv_path = os.path.join(os.path.dirname(__file__), 'temp_portfolio.csv')
    
    # Check if file exists to determine if we need headers
    portfolio_file_exists = os.path.exists(portfolio_csv_path)
    
    # Append to CSV file
    portfolio_df.to_csv(portfolio_csv_path, mode='a', header=not portfolio_file_exists, index=False)
    print(f"Detailed AS selection portfolio exported to {portfolio_csv_path}")
    print(f"Portfolio contains {len(portfolio_df)} protein-ligand pairs")
    print(f"AS selected algorithms distribution:")
    print(portfolio_df['selected_algorithm'].value_counts())
    
    # Generate detailed oracle selection portfolio
    oracle_portfolio_data = []
    
    for i in range(len(descriptive)):
        protein = descriptive.iloc[i]['protein']
        ligand = descriptive.iloc[i]['ligand']
        
        # Determine which algorithm provides the oracle performance
        # Filter only available algorithm columns that exist in the current dataset
        available_algos = [algo for algo in algorithm_columns if algo in descriptive.columns]
        
        # For scoring: find algorithm with maximum score
        algo_scores = descriptive.iloc[i][available_algos]
        # Convert to numeric, replacing non-numeric values with NaN
        algo_scores_numeric = pd.to_numeric(algo_scores, errors='coerce')
        # Find the algorithm with maximum score, handling NaN values
        if algo_scores_numeric.notna().any():
            oracle_algo_by_score = algo_scores_numeric.idxmax()
        else:
            oracle_algo_by_score = available_algos[0] if available_algos else 'surf'
        
        # For RMSD: find algorithm with minimum RMSD
        algo_rmsds = descriptive_rmsd.iloc[i][available_algos]
        # Convert to numeric, replacing non-numeric values with NaN
        algo_rmsds_numeric = pd.to_numeric(algo_rmsds, errors='coerce')
        # Find the algorithm with minimum RMSD, handling NaN values
        if algo_rmsds_numeric.notna().any():
            oracle_algo_by_rmsd = algo_rmsds_numeric.idxmin()
        else:
            oracle_algo_by_rmsd = available_algos[0] if available_algos else 'surf'
        
        # For pose validity: find any algorithm that passes pose validation
        algo_poses = descriptive_pose.iloc[i][available_algos]
        # Convert to numeric, replacing non-numeric values with NaN
        algo_poses_numeric = pd.to_numeric(algo_poses, errors='coerce')
        valid_algos = algo_poses_numeric[algo_poses_numeric == 1]
        oracle_algo_by_pose = valid_algos.index[0] if len(valid_algos) > 0 else (available_algos[0] if available_algos else 'surf')
        
        # Use the algorithm that provides best score as primary oracle choice
        oracle_algorithm = oracle_algo_by_score
        
        # Get oracle performance metrics
        if oracle_algorithm in descriptive.columns:
            oracle_score = descriptive.iloc[i][oracle_algorithm]
            oracle_pose_valid = descriptive_pose.iloc[i][oracle_algorithm]
            oracle_rmsd = descriptive_rmsd.iloc[i][oracle_algorithm]
        else:
            # Fallback to first available algorithm if oracle_algorithm doesn't exist
            fallback_algo = available_algos[0] if available_algos else 'surf'
            oracle_score = descriptive.iloc[i][fallback_algo] if fallback_algo in descriptive.columns else 0.0
            oracle_pose_valid = descriptive_pose.iloc[i][fallback_algo] if fallback_algo in descriptive_pose.columns else 0
            oracle_rmsd = descriptive_rmsd.iloc[i][fallback_algo] if fallback_algo in descriptive_rmsd.columns else 10.0
        
        # Get individual algorithm performance for comparison
        algo_performances = {}
        for algo in algorithm_columns:
            if algo in descriptive.columns:
                algo_performances[f'{algo}_score'] = descriptive.iloc[i][algo]
                algo_performances[f'{algo}_pose_valid'] = descriptive_pose.iloc[i][algo]
                algo_performances[f'{algo}_rmsd'] = descriptive_rmsd.iloc[i][algo]
        
        # Get prediction scores from the model for this case
        prediction_scores = {}
        for j, algo in enumerate(algorithm_columns):
            if j < len(cap_res_arr[i]):
                prediction_scores[f'{algo}_prediction'] = cap_res_arr[i][j]
        
        oracle_portfolio_entry = {
            'protein': protein,
            'ligand': ligand,
            'oracle_selected_algorithm': oracle_algorithm,
            'oracle_score': oracle_score,
            'oracle_pose_valid': oracle_pose_valid,
            'oracle_rmsd': oracle_rmsd,
            'oracle_rmsd_lt_1': 1 if oracle_rmsd < 1 else 0,
            'oracle_rmsd_lt_2': 1 if oracle_rmsd < 2 else 0,
            'oracle_rmsd_lt_1_valid': 1 if (oracle_rmsd < 1 and oracle_pose_valid) else 0,
            'oracle_rmsd_lt_2_valid': 1 if (oracle_rmsd < 2 and oracle_pose_valid) else 0,
            'oracle_algo_by_score': oracle_algo_by_score,
            'oracle_algo_by_rmsd': oracle_algo_by_rmsd,
            'oracle_algo_by_pose': oracle_algo_by_pose,
        }
        
        # Add individual algorithm performances
        oracle_portfolio_entry.update(algo_performances)
        # Add prediction scores
        oracle_portfolio_entry.update(prediction_scores)
        
        oracle_portfolio_data.append(oracle_portfolio_entry)
    
    # Create oracle portfolio DataFrame and save to CSV
    oracle_portfolio_df = pd.DataFrame(oracle_portfolio_data)
    oracle_portfolio_csv_path = os.path.join(os.path.dirname(__file__), 'temp_oracle_portfolio.csv')
    
    # Check if file exists to determine if we need headers
    oracle_portfolio_file_exists = os.path.exists(oracle_portfolio_csv_path)
    
    # Append to CSV file
    oracle_portfolio_df.to_csv(oracle_portfolio_csv_path, mode='a', header=not oracle_portfolio_file_exists, index=False)
    print(f"Detailed oracle selection portfolio exported to {oracle_portfolio_csv_path}")
    print(f"Oracle portfolio contains {len(oracle_portfolio_df)} protein-ligand pairs")
    print(f"Oracle selected algorithms distribution:")
    print(oracle_portfolio_df['oracle_selected_algorithm'].value_counts())
    
    # Now we have three complete dataframes, use them directly for evaluation
    scoring_columns = ['surf', 'uni', 'gnina', 'smina', 'qvina', 'diff', 'diffL', 'karma', 'AS', 'oracle', '2nd', '3rd']
    
    # Detailed metrics: RMSD < 1; RMSD < 2; Med RMSD; RMSD < 2 & Pose valid
    eval_results = dict()
    for col in scoring_columns:
        if col in descriptive.columns:
            # Extract values from respective dataframes
            rmsd_values = descriptive_rmsd[col]
            pose_values = descriptive_pose[col]
            
            # 1. Percent of RMSD < 1
            percent_rmsd_lt_1 = (rmsd_values < 1).mean() * 100
            # 2. Percent of RMSD < 2
            percent_rmsd_lt_2 = (rmsd_values < 2).mean() * 100                    
            # 3. Median RMSD
            median_rmsd = rmsd_values.median()
            # 4. Percent of RMSD < 2 & Pose valid
            percent_rmsd_lt_2_valid = ((rmsd_values < 2) & pose_values.astype(bool)).mean() * 100
            percent_rmsd_lt_1_valid = ((rmsd_values < 1) & pose_values.astype(bool)).mean() * 100
            # 5. Percent of RMSD < 5
            percent_rmsd_lt_5 = (rmsd_values < 5).mean() * 100   

            eval_results[col] = {
                'Percent RMSD < 1': round(percent_rmsd_lt_1, 2),
                'Percent RMSD < 2': round(percent_rmsd_lt_2, 2),
                'Percent RMSD < 5': round(percent_rmsd_lt_5, 2),
                'Median RMSD': round(median_rmsd, 3),
                'Percent RMSD < 1 (Valid Poses)': round(percent_rmsd_lt_1_valid, 2) if percent_rmsd_lt_1_valid is not None else 'N/A',
                'Percent RMSD < 2 (Valid Poses)': round(percent_rmsd_lt_2_valid, 2) if percent_rmsd_lt_2_valid is not None else 'N/A',
            }

    print(descriptive)
    print(descriptive.iloc[:,1:].sum(axis=0))

    print("Running statistical tests for individual cases: ")
    rmsd_gain = descriptive_rmsd['AS'].values - descriptive_rmsd['uni'].values
    pose_gain = descriptive_pose['AS'].values - descriptive_pose['uni'].values
    score_gain = descriptive['AS'].values - descriptive['uni'].values
    
    # Calculate baseline valid pose metrics for 'uni' method
    uni_rmsd_values = descriptive_rmsd['uni']
    uni_pose_values = descriptive_pose['uni']
    uni_percent_rmsd_lt_1_valid = ((uni_rmsd_values < 1) & uni_pose_values.astype(bool)).astype(int)
    uni_percent_rmsd_lt_2_valid = ((uni_rmsd_values < 2) & uni_pose_values.astype(bool)).astype(int)
    
    # Calculate AS valid pose metrics
    as_rmsd_values = descriptive_rmsd['AS']
    as_pose_values = descriptive_pose['AS']
    as_percent_rmsd_lt_1_valid = ((as_rmsd_values < 1) & as_pose_values.astype(bool)).astype(int)
    as_percent_rmsd_lt_2_valid = ((as_rmsd_values < 2) & as_pose_values.astype(bool)).astype(int)
    
    # Calculate gain metrics
    percent_rmsd_lt_1_valid_gain = as_percent_rmsd_lt_1_valid - uni_percent_rmsd_lt_1_valid
    percent_rmsd_lt_2_valid_gain = as_percent_rmsd_lt_2_valid - uni_percent_rmsd_lt_2_valid
    
    # Calculate AS valid pose metrics for comparison
    as_rmsd_values = descriptive_rmsd['AS']
    as_pose_values = descriptive_pose['AS']
    as_percent_rmsd_lt_1_valid_individual = ((as_rmsd_values < 1) & as_pose_values.astype(bool)).astype(int)
    as_percent_rmsd_lt_2_valid_individual = ((as_rmsd_values < 2) & as_pose_values.astype(bool)).astype(int)
    
    # Calculate gains for all algorithms
    algorithms = ['surf', 'uni', 'gnina', 'smina', 'qvina', 'diff', 'diffL', 'karma']
    gain_data_dict = {}
    
    # Add protein and ligand names to the data
    gain_data_dict['protein'] = descriptive['protein'].values
    gain_data_dict['ligand'] = descriptive['ligand'].values
    
    # Add raw AS performance data
    gain_data_dict['AS_rmsd_lt_1_valid'] = as_percent_rmsd_lt_1_valid_individual
    gain_data_dict['AS_rmsd_lt_2_valid'] = as_percent_rmsd_lt_2_valid_individual
    gain_data_dict['AS_rmsd'] = descriptive_rmsd['AS'].values
    gain_data_dict['AS_pose_valid'] = descriptive_pose['AS'].values.astype(int)
    gain_data_dict['AS_score'] = descriptive['AS'].values
    
    for algo in algorithms:
        if algo in descriptive_rmsd.columns and algo in descriptive_pose.columns:
            # Calculate baseline metrics for current algorithm
            algo_rmsd_values = descriptive_rmsd[algo]
            algo_pose_values = descriptive_pose[algo]
            algo_percent_rmsd_lt_1_valid = ((algo_rmsd_values < 1) & algo_pose_values.astype(bool)).astype(int)
            algo_percent_rmsd_lt_2_valid = ((algo_rmsd_values < 2) & algo_pose_values.astype(bool)).astype(int)
            
            # Calculate gains (AS - algorithm)
            gain_data_dict[f'{algo}_rmsd_lt_1_valid'] = algo_percent_rmsd_lt_1_valid
            gain_data_dict[f'{algo}_rmsd_lt_2_valid'] = algo_percent_rmsd_lt_2_valid
            gain_data_dict[f'gain_from_{algo}_rmsd_lt_1_valid'] = algo_percent_rmsd_lt_1_valid - as_percent_rmsd_lt_1_valid_individual
            gain_data_dict[f'gain_from_{algo}_rmsd_lt_2_valid'] = algo_percent_rmsd_lt_2_valid - as_percent_rmsd_lt_2_valid_individual

    # Export gain metrics to CSV file
    csv_file_path = os.path.join(os.path.dirname(__file__), 'temp.csv')
    
    # Create a DataFrame with all the gain metrics
    gain_data = pd.DataFrame(gain_data_dict)
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(csv_file_path)
    
    # Append to CSV file
    gain_data.to_csv(csv_file_path, mode='a', header=not file_exists, index=False)
    print(f"Performance gains for all algorithms exported to {csv_file_path}")
    print(f"Exported columns: {list(gain_data.columns)}")
    
    w_stat_rmsd, w_p_rmsd = stats.wilcoxon(rmsd_gain, alternative='greater')
    print(f"Wilcoxon signed-rank test for RMSD (gain > 0): statistic={w_stat_rmsd}, p-value={w_p_rmsd}.")
    w_stat_pose, w_p_pose = stats.wilcoxon(pose_gain, alternative='greater')
    print(f"Wilcoxon signed-rank test for PoseBuster (gain > 0): statistic={w_stat_pose}, p-value={w_p_pose}.")
    w_stat_score, w_p_score = stats.wilcoxon(score_gain, alternative='greater')
    print(f"Wilcoxon signed-rank test for Score (gain > 0): statistic={w_stat_score}, p-value={w_p_score}.")
    w_stat_1_valid, w_p_1_valid = stats.wilcoxon(percent_rmsd_lt_1_valid_gain, alternative='greater')
    print(f"Wilcoxon signed-rank test for RMSD<1 and PB pass (gain > 0): statistic={w_stat_1_valid}, p-value={w_p_1_valid}.")
    w_stat_2_valid, w_p_2_valid = stats.wilcoxon(percent_rmsd_lt_2_valid_gain, alternative='greater')
    print(f"Wilcoxon signed-rank test for RMSD<2 and PB pass (gain > 0): statistic={w_stat_2_valid}, p-value={w_p_2_valid}.")

    # Count occurrences of each algorithm in the 'choose' column (same for all dataframes)
    def count_selected_algorithms(descriptive_df):
        """
        Count the number of times each algorithm was selected.
        Args:
            descriptive_df: DataFrame containing the 'choose' column
        Returns:
            pandas.Series: Count of each algorithm selection
        """
        algorithm_counts = descriptive_df['choose'].value_counts()
        return algorithm_counts

    # Call the function and display results
    algorithm_counts = count_selected_algorithms(descriptive)
    print("\nAlgorithm Selection Counts:")
    print(algorithm_counts)
    # Optional: Display as percentages
    algorithm_percentages = (algorithm_counts / len(descriptive)) * 100
    print("\nAlgorithm Selection Percentages:")
    for algo, percentage in algorithm_percentages.items():
        print(f"{algo}: {percentage:.2f}%")

    # display evalutation results
    print("\nEvaluation Results:")
    for col, metrics in eval_results.items():
        print(f"\n{col}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    # plot_rmsd_metrics(eval_results)

if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    # LR Scheduler (not used in notebook, set to None)
    parser.add_argument('--lr_scheduler', default=None, type=str)
    parser.add_argument('--weight_decay', default=0, type=float)

    # Model and Data
    ## Option: 'binary', 'point'
    parser.add_argument('--model_name', default='binary', type=str)
    parser.add_argument('--loss', default='bce_with_logits', type=str)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    ## Controls for submodels
    ## Model 1 for ligands
    ## Options: GCN_L, GAT_L, GINE_L, SAGE
    parser.add_argument('--model1', default='GAT_L_fixed_out_dim', type=str)
    ## Feature dim
    parser.add_argument('--model1_features', default=25, type=int)
    ## Model 2 for proteins
    ## Options: GCN_GAT_GINE, GCN_GAT, GCN_GINE, GAT_GINE, GCN, GAT, GINE
    parser.add_argument('--model2', default='GCN_GAT_GINE_fixed_out_dim', type=str)
    ## Feature dim
    parser.add_argument('--model2_features', default=1280, type=int)
    ## Data related
    parser.add_argument('--drop_columns', default=[], type=str, nargs='*', help='Columns to drop (space-separated list)')
    parser.add_argument('--num_classes', default=8, type=int)  # This need to be changed with respect to the columns(algorithms) dropped
    parser.add_argument('--test_size', default=0.1, type=float)  # Test size for split
    ## KFold Support
    parser.add_argument('--k_folds', default=None, type=int)
    parser.add_argument('--fold_num', default=0, type=int)
    ## nDCG K config (eval)
    parser.add_argument('--ndcg_k', default=3, type=int, help='Rank cutoff for nDCG calculation')

    # Trainer args
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--accelerator', default='cuda', type=str)
    parser.add_argument('--devices', default=[0], type=int, nargs='+')

    # Test mode
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--ckpt_path', default=None, type=str, help='Path to the model checkpoint for testing')

    # NDCG loss parameters (train)
    parser.add_argument('--sigma', type=float, default=1.0)

    # Additional "rank" model parameters
    parser.add_argument('--use_ndcg_loss', action='store_true', default=True, help='Enable NDCG loss component')
    parser.add_argument('--no_ndcg_loss', dest='use_ndcg_loss', action='store_false', help='Disable NDCG loss component')
    parser.add_argument('--ndcg_loss_weight', type=float, default=1.0, help='Weight for NDCG loss component')
    
    parser.add_argument('--use_logistic_loss', action='store_true', default=False, help='Enable logistic loss component')
    parser.add_argument('--no_logistic_loss', dest='use_logistic_loss', action='store_false', help='Disable logistic loss component')
    parser.add_argument('--logistic_loss_weight', type=float, default=0.01, help='Weight for logistic loss component')

    # New arguments for separate test datasets
    parser.add_argument('--use_separate_test', action='store_true', 
                       help='Use separate test datasets instead of splitting from training data')
    parser.add_argument('--test_dataset_type', default='posebuster', choices=['posebuster', 'astex'], 
                       help='Which test dataset to use (posebuster or astex)')

    args = parser.parse_args()

    if args.test:
        test(args)
    else:
        main(args)