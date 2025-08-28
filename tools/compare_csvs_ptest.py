#!/usr/bin/env python3
"""
Script to perform statistical tests comparing baseline and improved versions
based on AS_rmsd_lt_1_valid and AS_rmsd_lt_2_valid metrics from two CSV files.

Usage:
    python compare_csvs_ptest.py baseline.csv improved.csv
    python compare_csvs_ptest.py --baseline baseline.csv --improved improved.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
import sys
import os


def load_and_validate_csv(file_path, file_type):
    """Load CSV file and validate required columns exist."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_type} data from {file_path}")
        print(f"  Shape: {df.shape}")
        
        # Check if required columns exist
        required_cols = ['AS_rmsd_lt_1_valid', 'AS_rmsd_lt_2_valid']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in {file_type}: {missing_cols}")
        
        print(f"  Required columns found: {required_cols}")
        return df
        
    except FileNotFoundError:
        print(f"Error: {file_type} file '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {file_type} file: {str(e)}")
        sys.exit(1)


def align_datasets(baseline_df, improved_df):
    """
    Align datasets based on protein-ligand pairs to ensure fair comparison.
    Returns aligned dataframes with matching rows.
    """
    # Create unique identifiers for each row
    if 'protein' in baseline_df.columns and 'ligand' in baseline_df.columns:
        baseline_df['pair_id'] = baseline_df['protein'].astype(str) + '_' + baseline_df['ligand'].astype(str)
        improved_df['pair_id'] = improved_df['protein'].astype(str) + '_' + improved_df['ligand'].astype(str)
        
        # Find common pairs
        common_pairs = set(baseline_df['pair_id']).intersection(set(improved_df['pair_id']))
        
        if len(common_pairs) == 0:
            print("Warning: No common protein-ligand pairs found between datasets.")
            print("Proceeding with row-by-row comparison assuming same order.")
            return baseline_df, improved_df
        
        # Filter to common pairs and sort for consistent ordering
        baseline_aligned = baseline_df[baseline_df['pair_id'].isin(common_pairs)].sort_values('pair_id')
        improved_aligned = improved_df[improved_df['pair_id'].isin(common_pairs)].sort_values('pair_id')
        
        print(f"Found {len(common_pairs)} common protein-ligand pairs for comparison.")
        
        return baseline_aligned, improved_aligned
    else:
        print("Warning: 'protein' and 'ligand' columns not found. Using row-by-row comparison.")
        min_length = min(len(baseline_df), len(improved_df))
        return baseline_df.iloc[:min_length], improved_df.iloc[:min_length]


def perform_statistical_tests(baseline_values, improved_values, metric_name):
    """Perform various statistical tests comparing baseline and improved values."""
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(baseline_values) | np.isnan(improved_values))
    baseline_clean = baseline_values[valid_mask]
    improved_clean = improved_values[valid_mask]
    
    if len(baseline_clean) == 0:
        print(f"  {metric_name}: No valid data points for comparison")
        return
    
    print(f"\n--- {metric_name.upper().replace('_', ' ')} ---")
    print(f"Valid data points: {len(baseline_clean)}")
    
    # Calculate basic statistics
    baseline_mean = np.mean(baseline_clean)
    improved_mean = np.mean(improved_clean)
    baseline_std = np.std(baseline_clean)
    improved_std = np.std(improved_clean)
    
    # Calculate improvement statistics
    improvements = improved_clean - baseline_clean
    mean_improvement = np.mean(improvements)
    positive_improvements = np.sum(improvements > 0)
    negative_improvements = np.sum(improvements < 0)
    no_change = np.sum(improvements == 0)
    
    print(f"Baseline - Mean: {baseline_mean:.3f}, Std: {baseline_std:.3f}")
    print(f"Improved - Mean: {improved_mean:.3f}, Std: {improved_std:.3f}")
    print(f"Mean improvement: {mean_improvement:.3f}")
    print(f"Improvements: +{positive_improvements}, -{negative_improvements}, ={no_change}")
    
    # 1. Paired t-test (parametric)
    try:
        t_stat, t_p = stats.ttest_rel(improved_clean, baseline_clean, alternative='greater')
        print(f"Paired t-test:")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {t_p:.6f} {get_significance(t_p)}")
    except Exception as e:
        print(f"Paired t-test failed: {str(e)}")
    
    # 2. Wilcoxon signed-rank test (non-parametric)
    try:
        if np.any(improvements != 0):  # Check if there are non-zero differences
            w_stat, w_p = stats.wilcoxon(improved_clean, baseline_clean, alternative='greater')
            print(f"Wilcoxon signed-rank test:")
            print(f"  W-statistic: {w_stat:.4f}, p-value: {w_p:.6f} {get_significance(w_p)}")
        else:
            print("Wilcoxon signed-rank test: All differences are zero")
    except Exception as e:
        print(f"Wilcoxon signed-rank test failed: {str(e)}")
    
    # 3. Mann-Whitney U test (independent samples)
    try:
        u_stat, u_p = stats.mannwhitneyu(improved_clean, baseline_clean, alternative='greater')
        print(f"Mann-Whitney U test:")
        print(f"  U-statistic: {u_stat:.4f}, p-value: {u_p:.6f} {get_significance(u_p)}")
    except Exception as e:
        print(f"Mann-Whitney U test failed: {str(e)}")
    
    # 4. McNemar's test (for binary outcomes)
    if set(baseline_clean).issubset({0, 1}) and set(improved_clean).issubset({0, 1}):
        try:
            # Create contingency table
            n_00 = np.sum((baseline_clean == 0) & (improved_clean == 0))
            n_01 = np.sum((baseline_clean == 0) & (improved_clean == 1))
            n_10 = np.sum((baseline_clean == 1) & (improved_clean == 0))
            n_11 = np.sum((baseline_clean == 1) & (improved_clean == 1))
            
            print(f"Contingency table:")
            print(f"  Baseline=0, Improved=0: {n_00}")
            print(f"  Baseline=0, Improved=1: {n_01}")
            print(f"  Baseline=1, Improved=0: {n_10}")
            print(f"  Baseline=1, Improved=1: {n_11}")
            
            # McNemar's test
            mcnemar_stat = (abs(n_01 - n_10) - 1)**2 / (n_01 + n_10) if (n_01 + n_10) > 0 else 0
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
            
            print(f"McNemar's test:")
            print(f"  Chi-square: {mcnemar_stat:.4f}, p-value: {mcnemar_p:.6f} {get_significance(mcnemar_p)}")
            
        except Exception as e:
            print(f"McNemar's test failed: {str(e)}")
    
    # 5. Effect size (Cohen's d)
    try:
        pooled_std = np.sqrt((baseline_std**2 + improved_std**2) / 2)
        cohens_d = (improved_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
        print(f"Effect size (Cohen's d): {cohens_d:.4f} {interpret_cohens_d(cohens_d)}")
    except Exception as e:
        print(f"Effect size calculation failed: {str(e)}")


def get_significance(p_value):
    """Return significance indicator based on p-value."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "(negligible)"
    elif abs_d < 0.5:
        return "(small)"
    elif abs_d < 0.8:
        return "(medium)"
    else:
        return "(large)"


def main():
    parser = argparse.ArgumentParser(
        description="Perform statistical tests comparing baseline and improved versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_csvs_ptest.py baseline.csv improved.csv
  python compare_csvs_ptest.py --baseline baseline.csv --improved improved.csv
        """
    )
    
    parser.add_argument('baseline', nargs='?', help='Path to baseline CSV file')
    parser.add_argument('improved', nargs='?', help='Path to improved CSV file')
    parser.add_argument('--baseline', dest='baseline_file', help='Path to baseline CSV file')
    parser.add_argument('--improved', dest='improved_file', help='Path to improved CSV file')
    parser.add_argument('--output', help='Output file to save results (optional)')
    
    args = parser.parse_args()
    
    # Determine file paths
    baseline_path = args.baseline or args.baseline_file
    improved_path = args.improved or args.improved_file
    
    if not baseline_path or not improved_path:
        print("Error: Both baseline and improved CSV files must be specified.")
        parser.print_help()
        sys.exit(1)
    
    # Load and validate data
    baseline_df = load_and_validate_csv(baseline_path, "baseline")
    improved_df = load_and_validate_csv(improved_path, "improved")
    
    # Align datasets
    baseline_aligned, improved_aligned = align_datasets(baseline_df, improved_df)
    
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON RESULTS")
    print("="*80)
    print(f"Comparing: {os.path.basename(baseline_path)} vs {os.path.basename(improved_path)}")
    
    # Perform tests for each metric
    metrics = ['AS_rmsd_lt_1_valid', 'AS_rmsd_lt_2_valid']
    
    for metric in metrics:
        baseline_values = baseline_aligned[metric].values
        improved_values = improved_aligned[metric].values
        
        perform_statistical_tests(baseline_values, improved_values, metric)
    
    print("\n" + "="*80)
    print("LEGEND:")
    print("*** p<0.001 (highly significant)")
    print("**  p<0.01  (very significant)")
    print("*   p<0.05  (significant)")
    print("Effect sizes: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)")
    print("="*80)
    
    # Save results if output file specified
    if args.output:
        try:
            # You could implement saving detailed results to file here
            print(f"\nResults would be saved to: {args.output}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")


if __name__ == "__main__":
    main()
