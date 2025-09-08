#!/usr/bin/env python3
# Statistical comparison of two models using stats.py

import argparse
import pickle
import numpy as np
import os
import sys

# Add the path to your stats.py
sys.path.append('local/sandbox/ci_stat_test/')

# Import your existing statistical functions
try:
    from stats import bootstrap_t_test, t_test, bootstrap_ci
    STATS_AVAILABLE = True
    print("Using custom stats.py functions")
except ImportError:
    STATS_AVAILABLE = False
    print("Warning: stats.py not found, using basic functions only")

def parse_args():
    parser = argparse.ArgumentParser(description="Compare two models statistically using stats.py")
    parser.add_argument("--model1_results", type=str, required=True,
                        help="Pickle file with model 1 results")
    parser.add_argument("--model2_results", type=str, required=True,
                        help="Pickle file with model 2 results")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                        help="Directory to save comparison results")
    parser.add_argument("--n_bootstrap", type=int, default=10000,
                        help="Number of bootstrap samples for tests")
    
    return parser.parse_args()

def load_results(filepath):
    """Load results from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

def mcnemar_test_manual(results1, results2):
    """Manual McNemar's test implementation"""
    # Get raw predictions
    signal1 = np.array(results1['raw_results']['signal_correct'])
    noise1 = np.array(results1['raw_results']['noise_correct'])
    signal2 = np.array(results2['raw_results']['signal_correct'])
    noise2 = np.array(results2['raw_results']['noise_correct'])
    
    # Combine into overall correctness
    all_correct1 = np.concatenate([signal1, noise1])
    all_correct2 = np.concatenate([signal2, noise2])
    
    # Create contingency table
    both_correct = np.sum(all_correct1 & all_correct2)
    model1_only = np.sum(all_correct1 & ~all_correct2)
    model2_only = np.sum(~all_correct1 & all_correct2)
    both_wrong = np.sum(~all_correct1 & ~all_correct2)
    
    print("McNemar's Test Contingency Table:")
    print("                    Model 2")
    print("                Correct  Wrong")
    print(f"Model 1 Correct   {both_correct:6d}  {model1_only:5d}")
    print(f"        Wrong     {model2_only:6d}  {both_wrong:5d}")
    print()
    
    # McNemar's test statistic
    if model1_only + model2_only > 0:
        chi_square = (abs(model1_only - model2_only) - 1)**2 / (model1_only + model2_only)
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(chi_square, df=1)
        return chi_square, p_value
    else:
        return 0, 1.0

def compare_metrics_with_stats_py(results1, results2, metric_name, n_bootstrap=10000):
    """Compare specific metrics using bootstrap t-test from stats.py"""
    if not STATS_AVAILABLE:
        print(f"stats.py not available, skipping bootstrap t-test for {metric_name}")
        return None
    
    # Get bootstrap samples for the metric
    values1 = np.array(results1['bootstrap_results'][metric_name])
    values2 = np.array(results2['bootstrap_results'][metric_name])
    
    print(f"\nBootstrap t-test for {metric_name}:")
    print(f"Model 1 mean: {np.mean(values1):.4f}")
    print(f"Model 2 mean: {np.mean(values2):.4f}")
    print(f"Difference: {np.mean(values1) - np.mean(values2):.4f}")
    
    # Perform bootstrap t-test (two-tailed)
    try:
        result_two_tailed = bootstrap_t_test(values1, values2, paired=False, bootnum=n_bootstrap)
        print(f"Two-tailed p-value: {result_two_tailed['p-value']:.4f}")
        
        # One-tailed test (greater)
        result_greater = bootstrap_t_test(values1, values2, paired=False, alternative="greater", bootnum=n_bootstrap)
        print(f"One-tailed p-value (Model 1 > Model 2): {result_greater['p-value']:.4f}")
        
        return {
            'two_tailed_p': result_two_tailed['p-value'],
            'greater_p': result_greater['p-value'],
            'mean_diff': np.mean(values1) - np.mean(values2)
        }
    except Exception as e:
        print(f"Error in bootstrap t-test: {e}")
        return None

def save_comparison_report(results1, results2, output_dir, mcnemar_stat, mcnemar_p, bootstrap_tests):
    """Save detailed comparison report"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "comparison_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("Statistical Comparison Report (using stats.py)\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Model 1: {results1['model_name']}\n")
        f.write(f"Model 2: {results2['model_name']}\n\n")
        
        f.write("Basic Metrics Comparison:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Metric':<15} {'Model 1':<10} {'Model 2':<10} {'Difference':<12}\n")
        f.write("-"*50 + "\n")
        
        for metric in ['accuracy_total', 'precision', 'recall', 'f_score']:
            val1 = results1['metrics'][metric]
            val2 = results2['metrics'][metric]
            diff = val1 - val2
            f.write(f"{metric:<15} {val1:<10.4f} {val2:<10.4f} {diff:<12.4f}\n")
        
        f.write(f"\nMcNemar's Test Results:\n")
        f.write("-"*30 + "\n")
        f.write(f"Chi-square statistic: {mcnemar_stat:.4f}\n")
        f.write(f"p-value: {mcnemar_p:.4f}\n")
        
        if mcnemar_p < 0.05:
            f.write("Result: SIGNIFICANT DIFFERENCE (p < 0.05)\n")
        else:
            f.write("Result: No significant difference (p >= 0.05)\n")
        
        f.write(f"\nBootstrap t-test Results (using stats.py):\n")
        f.write("-"*50 + "\n")
        
        for metric, test_result in bootstrap_tests.items():
            if test_result:
                f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                f.write(f"  Mean difference: {test_result['mean_diff']:.4f}\n")
                f.write(f"  Two-tailed p-value: {test_result['two_tailed_p']:.4f}\n")
                f.write(f"  One-tailed p-value (Model 1 > Model 2): {test_result['greater_p']:.4f}\n")
                
                if test_result['two_tailed_p'] < 0.05:
                    f.write("  Result: SIGNIFICANT DIFFERENCE (two-tailed)\n")
                else:
                    f.write("  Result: No significant difference (two-tailed)\n")
    
    print(f"Detailed report saved: {report_path}")

def main():
    args = parse_args()
    
    # Load results
    print("Loading results...")
    results1 = load_results(args.model1_results)
    results2 = load_results(args.model2_results)
    
    model1_name = results1['model_name']
    model2_name = results2['model_name']
    
    print(f"Comparing {model1_name} vs {model2_name}")
    print("="*60)
    
    # Basic comparison
    print("\nBasic Metrics Comparison:")
    print(f"{'Metric':<15} {'Model 1':<10} {'Model 2':<10} {'Difference':<12}")
    print("-" * 50)
    
    for metric in ['accuracy_total', 'precision', 'recall', 'f_score']:
        val1 = results1['metrics'][metric]
        val2 = results2['metrics'][metric]
        diff = val1 - val2
        print(f"{metric:<15} {val1:<10.4f} {val2:<10.4f} {diff:<12.4f}")
    
    # McNemar's test
    print(f"\n{'='*60}")
    print("McNemar's Test (Overall Classification Accuracy)")
    print("="*60)
    
    mcnemar_stat, mcnemar_p = mcnemar_test_manual(results1, results2)
    
    print(f"McNemar's χ² statistic: {mcnemar_stat:.4f}")
    print(f"p-value: {mcnemar_p:.4f}")
    
    if mcnemar_p < 0.05:
        print("*** SIGNIFICANT DIFFERENCE (p < 0.05) ***")
    else:
        print("No significant difference (p ≥ 0.05)")
    
    # Bootstrap t-tests using stats.py
    print(f"\n{'='*60}")
    print("Bootstrap t-tests (using stats.py)")
    print("="*60)
    
    metrics_to_test = ['accuracy_total', 'precision', 'recall', 'f_score']
    bootstrap_tests = {}
    
    for metric in metrics_to_test:
        bootstrap_tests[metric] = compare_metrics_with_stats_py(
            results1, results2, metric, args.n_bootstrap)
    
    # Save detailed report
    save_comparison_report(results1, results2, args.output_dir, 
                         mcnemar_stat, mcnemar_p, bootstrap_tests)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    
    significant_tests = []
    if mcnemar_p < 0.05:
        significant_tests.append("McNemar's test")
    
    for metric, test_result in bootstrap_tests.items():
        if test_result and test_result['two_tailed_p'] < 0.05:
            significant_tests.append(f"Bootstrap t-test {metric}")
    
    if significant_tests:
        print(f"Significant differences found in: {', '.join(significant_tests)}")
        print(f"{model1_name} vs {model2_name}: STATISTICALLY DIFFERENT")
    else:
        print("No significant differences found in any test")
        print(f"{model1_name} vs {model2_name}: NOT STATISTICALLY DIFFERENT")
    
    print(f"\nResults saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()
