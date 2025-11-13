# MC-GNNAS-Dock

Official implementation of PRICAI 25 conference paper - MC-GNNAS-Dock: Multi-criteria GNN-based Algorithm Selection for Molecular Docking

## Use

(The documentation is still under construction.)

### Installation

We have tested the code on Ubuntu 22.04 and 24.04 with CUDA 12.4 and Python 3.8.

```bash
conda env create -f environment.yml 
conda activate GNN_AS
pip install -r requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### Data Preparation

Download the dataset from [here](https://duke.box.com/s/pp7740u2qkxiazy45nir44esbiyykhqh)

Create a folder named `dataset` for the dataset, which includes:
- ligand_g_mod/ (ligand graphs)
- protein_g_mod/ (protein graphs) (Consider using symbol links for flexibility with storage!)
- moad_pose_check(no_rmsd).csv (PoseBuster check results for targetted algorithms)
- rmsd_train.csv (RMSD results for targetted algorithms)

### Training

```bash
./train_kfold.sh
```

#### Parameters

(Default)
```bash
python main.py --k_folds 10 --fold_num $i --model_name rank --max_epochs 200 --loss bce --devices 0 --seed 42
```

You may config the weight of the 2 ranking aware losses seperately with:
```bash
# None
python main.py --k_folds 10 --fold_num $i --model_name rank --max_epochs 200 --loss bce --devices 0 --seed 42 --no_ndcg_loss --no_logistic_loss
# LambdaNDCG Loss
python main.py --k_folds 10 --fold_num $i --model_name rank --max_epochs 200 --loss bce --devices 0 --seed 42 --use_ndcg_loss --ndcg_loss_weight 2.0 --no_logistic_loss 
# PairWise Logistic Loss
python main.py --k_folds 10 --fold_num $i --model_name rank --max_epochs 200 --loss bce --devices 0 --seed 42 --no_ndcg_loss --use_logistic_loss --logistic_loss_weight 0.02 
# Both
python main.py --k_folds 10 --fold_num $i --model_name rank --max_epochs 200 --loss bce --devices 0 --seed 42 --use_ndcg_loss --ndcg_loss_weight 2.0 --use_logistic_loss --logistic_loss_weight 0.02
```

You may also train with a subset of the full algorithm portfolio (Here we drop 4 algorithms, therefore the remaining `num_classes` is $8 - 4 = 4$):
```bash
# None
python main.py --k_folds 10 --fold_num $i --model_name rank --max_epochs 200 --loss bce --devices 0 --seed 42 --no_ndcg_loss --no_logistic_loss --drop_columns smina diff diffL karma --num_classes 4
# LambdaNDCG Loss
python main.py --k_folds 10 --fold_num $i --model_name rank --max_epochs 200 --loss bce --devices 0 --seed 42 --use_ndcg_loss --ndcg_loss_weight 2.0 --no_logistic_loss --drop_columns smina diff diffL karma --num_classes 4
# PairWise Logistic Loss
python main.py --k_folds 10 --fold_num $i --model_name rank --max_epochs 200 --loss bce --devices 0 --seed 42 --no_ndcg_loss --use_logistic_loss --logistic_loss_weight 0.02 --drop_columns smina diff diffL karma --num_classes 4
# Both
python main.py --k_folds 10 --fold_num $i --model_name rank --max_epochs 200 --loss bce --devices 0 --seed 42 --use_ndcg_loss --ndcg_loss_weight 2.0 --use_logistic_loss --logistic_loss_weight 0.02 --drop_columns smina diff diffL karma --num_classes 4
```

### Evaluation

```bash
./test_kfold.sh
```
If using a subset of the full algorithm portfolio, please add the flags `--drop_columns` `--num_classes`

Use the notebook `tools/result_analysis.ipynb` to analyze the results.

## Citation

If you find our work useful for your research, please consider citing our paper:

<a id="1">[1]</a>  S. Cao, H. Wu, J. B. Wang, Y. Yuan, and M. Misir, “MC-GNNAS-Dock: Multi-criteria GNN-based Algorithm Selection for Molecular Docking,” Oct. 01, 2025, arXiv: arXiv:2509.26377. doi: 10.48550/arXiv.2509.26377.
