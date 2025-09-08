#!/usr/bin/env python3
# Enhanced evaluation script with statistical analysis using existing stats.py
# Based on original implementation by bin-wu

import os
import re
import argparse
import json
import numpy as np
from scipy import stats
import pickle
from collections import defaultdict
import sys

# Add the path to your stats.py if needed
sys.path.append('local/sandbox/ci_stat_test/')

# Import your existing statistical functions
try:
    from stats import bootstrap_ci, bootstrap_t_test, t_test
    STATS_AVAILABLE = True
    print("Using custom stats.py functions")
except ImportError:
    STATS_AVAILABLE = False
    print("Warning: stats.py not found, using basic functions only")

classes = ['chi','cha','ek','noise','ot','ph','tr','trph','ts','tw']

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute accuracy, F-score, confidence intervals, and statistical tests",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--hypo_files", type=str, nargs="+", required=True,
                        help='Prediction files in Audacity label format')
    parser.add_argument("--ref_files", type=str, nargs="+", required=True,
                        help='Reference files in Audacity label format')
    parser.add_argument("--model_name", type=str, default="Model",
                        help='Name of the model for output')
    parser.add_argument("--save_results", type=str, default=None,
                        help='Path to save detailed results (pickle file)')
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help='Number of bootstrap samples')
    parser.add_argument("--confidence", type=float, default=0.95,
                        help='Confidence level for intervals')
    
    return parser.parse_args()

def compare_segments(prediction_list, correct_list, classes):
    """Compare segments and return detailed results"""
    noise_correct = []
    signal_correct = []
    non_noise_correct = []
    
    # Store per-file results for more detailed analysis
    per_file_results = []
    
    for i, (pred_file, corr_file) in enumerate(zip(prediction_list, correct_list)):
        print(f"Processing file pair {i+1}/{len(prediction_list)}: {os.path.basename(pred_file)}")
        
        with open(pred_file, 'r') as predictions, open(corr_file, 'r') as correct:
            lines_pred = discretize_segments(predictions, classes)
            lines_corr, first, last = discretize_segments(correct, classes, return_indices=True)

        # Extend prediction if shorter
        lines_pred.extend(['noise'] * (len(lines_corr) - len(lines_pred)))
        
        # Per-file statistics
        file_noise_correct = []
        file_signal_correct = []
        file_non_noise_correct = []
        
        for pred_label, corr_label in zip(lines_pred[first:last], lines_corr[first:last]):
            if corr_label == 'noise':
                is_correct = pred_label == 'noise'
                noise_correct.append(is_correct)
                file_noise_correct.append(is_correct)
            else:
                is_signal_correct = pred_label == corr_label
                is_non_noise_correct = pred_label != 'noise'
                
                signal_correct.append(is_signal_correct)
                non_noise_correct.append(is_non_noise_correct)
                file_signal_correct.append(is_signal_correct)
                file_non_noise_correct.append(is_non_noise_correct)
        
        per_file_results.append({
            'file': os.path.basename(pred_file),
            'noise_correct': file_noise_correct,
            'signal_correct': file_signal_correct,
            'non_noise_correct': file_non_noise_correct
        })

    return noise_correct, signal_correct, non_noise_correct, per_file_results

def discretize_segments(file, classes, return_indices=False):
    """Discretize segments into 50ms chunks"""
    lines = []
    current = 0
    start_window = current * 0.05
    middle_start = start_window + 0.225
    middle_end = start_window + 0.275
    first = None

    for line in file:
        start_t, end_t, label = re.split(r'\s+', line.strip())
        start_t, end_t = float(start_t), float(end_t)
        label = label.lower()

        while start_t - middle_end > -0.001:
            lines.append('noise')
            current += 1
            start_window = current * 0.05
            middle_start = start_window + 0.225
            middle_end = start_window + 0.275

        while end_t - middle_start > 0.001:
            if first is None:
                first = current
            lines.append('noise' if label not in classes else label)
            current += 1
            start_window = current * 0.05
            middle_start = start_window + 0.225
            middle_end = start_window + 0.275

    last = len(lines)

    if return_indices:
        return lines, first, last
    else:
        return lines

def bootstrap_metrics_with_stats_py(signal_correct, noise_correct, non_noise_correct, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence intervals using stats.py functions"""
    if not STATS_AVAILABLE:
        print("stats.py not available, using basic bootstrap")
        return None
        # return bootstrap_metrics_basic(signal_correct, noise_correct, non_noise_correct, n_bootstrap, confidence)
    
    print(f"Computing bootstrap confidence intervals using stats.py ({n_bootstrap} samples)...")
    
    signal_correct = np.array(signal_correct, dtype=float)
    noise_correct = np.array(noise_correct, dtype=float)
    non_noise_correct = np.array(non_noise_correct, dtype=float)
    
    # Calculate metrics for individual bootstrap samples
    all_correct = np.concatenate([signal_correct, noise_correct])
    
    # Use your bootstrap_ci function for each metric
    alpha = 1 - confidence
    
    # Bootstrap confidence intervals for each metric
    accuracy_total_ci = bootstrap_ci(all_correct, bootnum=n_bootstrap, bootfunc=np.mean, alpha=alpha)
    accuracy_signal_ci = bootstrap_ci(signal_correct, bootnum=n_bootstrap, bootfunc=np.mean, alpha=alpha)
    accuracy_noise_ci = bootstrap_ci(noise_correct, bootnum=n_bootstrap, bootfunc=np.mean, alpha=alpha)
    accuracy_non_noise_ci = bootstrap_ci(non_noise_correct, bootnum=n_bootstrap, bootfunc=np.mean, alpha=alpha)
    
    # For precision, recall, F1 - need custom bootstrap function
    def precision_func(signal_arr):
        tp = np.sum(signal_arr)
        # For bootstrap, we need to estimate fp from the noise_correct data
        fp_rate = 1 - np.mean(noise_correct)  # Use original noise accuracy
        fp = fp_rate * len(noise_correct)
        return tp / (tp + fp) if tp + fp > 0 else 0
    
    def recall_func(signal_arr):
        return np.mean(signal_arr)
    
    def f1_func(signal_arr):
        recall = np.mean(signal_arr)
        tp = np.sum(signal_arr)
        fp_rate = 1 - np.mean(noise_correct)
        fp = fp_rate * len(noise_correct)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    precision_ci = bootstrap_ci(signal_correct, bootnum=n_bootstrap, bootfunc=precision_func, alpha=alpha)
    recall_ci = bootstrap_ci(signal_correct, bootnum=n_bootstrap, bootfunc=recall_func, alpha=alpha)
    f1_ci = bootstrap_ci(signal_correct, bootnum=n_bootstrap, bootfunc=f1_func, alpha=alpha)
    
    confidence_intervals = {
        'accuracy_total': tuple(accuracy_total_ci['ci']),
        'accuracy_signal': tuple(accuracy_signal_ci['ci']),
        'accuracy_noise': tuple(accuracy_noise_ci['ci']),
        'accuracy_non_noise': tuple(accuracy_non_noise_ci['ci']),
        'precision': tuple(precision_ci['ci']),
        'recall': tuple(recall_ci['ci']),
        'f_score': tuple(f1_ci['ci'])
    }
    
    bootstrap_results = {
        'accuracy_total': accuracy_total_ci['stat'],
        'accuracy_signal': accuracy_signal_ci['stat'],
        'accuracy_noise': accuracy_noise_ci['stat'],
        'accuracy_non_noise': accuracy_non_noise_ci['stat'],
        'precision': precision_ci['stat'],
        'recall': recall_ci['stat'],
        'f_score': f1_ci['stat']
    }
    
    return confidence_intervals, bootstrap_results

def bootstrap_metrics_basic(signal_correct, noise_correct, non_noise_correct, n_bootstrap=1000, confidence=0.95):
    """Basic bootstrap implementation as fallback"""
    print(f"Computing bootstrap confidence intervals (basic method, {n_bootstrap} samples)...")
    
    signal_correct = np.array(signal_correct)
    noise_correct = np.array(noise_correct)
    non_noise_correct = np.array(non_noise_correct)
    
    bootstrap_results = {
        'accuracy_noise': [],
        'accuracy_signal': [],
        'accuracy_total': [],
        'accuracy_non_noise': [],
        'precision': [],
        'recall': [],
        'f_score': []
    }
    
    for i in range(n_bootstrap):
        if i % 200 == 0:
            print(f"Bootstrap sample {i+1}/{n_bootstrap}")
        
        # Bootstrap sample indices
        signal_indices = np.random.choice(len(signal_correct), size=len(signal_correct), replace=True)
        noise_indices = np.random.choice(len(noise_correct), size=len(noise_correct), replace=True)
        
        # Bootstrap samples
        boot_signal = signal_correct[signal_indices]
        boot_noise = noise_correct[noise_indices]
        boot_non_noise = non_noise_correct[signal_indices]
        
        # Calculate metrics
        acc_noise = np.mean(boot_noise)
        acc_signal = np.mean(boot_signal)
        acc_total = np.mean(np.concatenate([boot_signal, boot_noise]))
        acc_non_noise = np.mean(boot_non_noise)
        
        # Precision, recall, F1
        tp = np.sum(boot_signal)
        fp = np.sum(~boot_noise)
        fn = np.sum(~boot_signal)
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        bootstrap_results['accuracy_noise'].append(acc_noise)
        bootstrap_results['accuracy_signal'].append(acc_signal)
        bootstrap_results['accuracy_total'].append(acc_total)
        bootstrap_results['accuracy_non_noise'].append(acc_non_noise)
        bootstrap_results['precision'].append(precision)
        bootstrap_results['recall'].append(recall)
        bootstrap_results['f_score'].append(f_score)
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    confidence_intervals = {}
    for metric, values in bootstrap_results.items():
        lower = np.percentile(values, 100 * alpha / 2)
        upper = np.percentile(values, 100 * (1 - alpha / 2))
        confidence_intervals[metric] = (lower, upper)
    
    return confidence_intervals, bootstrap_results

def evaluate_metrics(signal_correct, noise_correct, non_noise_correct):
    """Calculate basic metrics"""
    noise_correct = np.array(noise_correct)
    signal_correct = np.array(signal_correct)
    non_noise_correct = np.array(non_noise_correct)

    accuracy_noise = np.mean(noise_correct)
    accuracy_signal = np.mean(signal_correct)
    accuracy_total = np.mean(np.concatenate((signal_correct, noise_correct)))
    accuracy_non_noise = np.mean(non_noise_correct)

    true_positives = np.sum(signal_correct)
    false_positives = np.sum(~noise_correct)
    false_negatives = np.sum(~signal_correct)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "accuracy_noise": accuracy_noise,
        "accuracy_signal": accuracy_signal,
        "accuracy_total": accuracy_total,
        "accuracy_non_noise": accuracy_non_noise,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def print_results_with_ci(metrics, confidence_intervals, model_name, confidence=0.95):
    """Print results with confidence intervals"""
    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")
    
    print("Basic Metrics:")
    print(f"Fraction correctly classified: Noise:{metrics['accuracy_noise']:.4f}, Call:{metrics['accuracy_signal']:.4f}, Total:{metrics['accuracy_total']:.4f}")
    print(f"Caller accuracy:{metrics['accuracy_non_noise']:.4f}")
    print(f"Recall:{metrics['recall']:.4f}, Precision:{metrics['precision']:.4f}, F1-score:{metrics['f_score']:.4f}")
    
    print(f"\n{confidence*100:.0f}% Confidence Intervals:")
    for metric_name, (lower, upper) in confidence_intervals.items():
        metric_value = metrics[metric_name]
        print(f"{metric_name:15}: {metric_value:.4f} [{lower:.4f}, {upper:.4f}]")
    
    print(f"\nConfusion Matrix Elements:")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")

def main():
    args = parse_args()
    
    print(f"Evaluating {args.model_name}")
    print(f"Prediction files: {args.hypo_files}")
    print(f"Reference files: {args.ref_files}")
    
    # Check if files exist
    for file_path in args.hypo_files + args.ref_files:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return None, None, None
    
    # Main evaluation
    noise_correct, signal_correct, non_noise_correct, per_file_results = compare_segments(
        args.hypo_files, args.ref_files, classes)
    
    # Calculate basic metrics
    metrics = evaluate_metrics(signal_correct, noise_correct, non_noise_correct)
    
    # Bootstrap confidence intervals
    confidence_intervals, bootstrap_results = bootstrap_metrics_with_stats_py(
        signal_correct, noise_correct, non_noise_correct, 
        args.n_bootstrap, args.confidence)
    
    # Print results
    print_results_with_ci(metrics, confidence_intervals, args.model_name, args.confidence)
    
    # Save detailed results if requested
    if args.save_results:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
        
        results_data = {
            'model_name': args.model_name,
            'metrics': metrics,
            'confidence_intervals': confidence_intervals,
            'bootstrap_results': bootstrap_results,
            'raw_results': {
                'noise_correct': noise_correct,
                'signal_correct': signal_correct,
                'non_noise_correct': non_noise_correct
            },
            'per_file_results': per_file_results,
            'args': vars(args)
        }
        
        with open(args.save_results, 'wb') as f:
            pickle.dump(results_data, f)
        print(f"\nDetailed results saved to: {args.save_results}")
    
    return metrics, confidence_intervals, bootstrap_results

if __name__ == "__main__":
    main()
