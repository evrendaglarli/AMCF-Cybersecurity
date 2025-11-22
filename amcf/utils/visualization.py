"""Visualization utilities for results"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def plot_recall_comparison(results: Dict[str, Dict[float, float]],
                          save_path: str = None):
    """
    Plot Recall@B comparison across models and budgets
    
    Args:
        results: {model: {budget: recall}}
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    budgets = sorted(list(results[list(results.keys())[0]].keys()))
    
    for model, budget_recalls in results.items():
        recalls = [budget_recalls[b] for b in budgets]
        ax.plot(budgets, recalls, marker='o', label=model, linewidth=2)
    
    ax.set_xlabel('False Positive Budget (FP/hour)', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Detection Performance: Recall@B', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_operational_metrics(metrics_df, save_path: str = None):
    """Plot operational metrics comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Long-dwell recall
    ax = axes[0, 0]
    models = metrics_df['model'].unique()
    dwell_recalls = [metrics_df[metrics_df['model'] == m]['long_dwell_recall'].values[0] for m in models]
    ax.bar(models, dwell_recalls, color='steelblue')
    ax.set_ylabel('Long-Dwell Recall', fontsize=11)
    ax.set_title('Multi-Stage Attack Detection')
    ax.set_ylim([0, 1])
    
    # Alerts per hour
    ax = axes[0, 1]
    alerts = [metrics_df[metrics_df['model'] == m]['alerts_per_hour'].values[0] for m in models]
    ax.bar(models, alerts, color='coral')
    ax.set_ylabel('Alerts/Hour', fontsize=11)
    ax.set_title('Alert Volume at 0.80 Recall')
    
    # TTC
    ax = axes[1, 0]
    ttc = [metrics_df[metrics_df['model'] == m]['ttc'].values[0] for m in models]
    ax.bar(models, ttc, color='seagreen')
    ax.set_ylabel('Median TTC (minutes)', fontsize=11)
    ax.set_title('Time-to-Containment')
    
    # Escalation precision
    ax = axes[1, 1]
    precision = [metrics_df[metrics_df['model'] == m]['precision'].values[0] for m in models]
    ax.bar(models, precision, color='mediumpurple')
    ax.set_ylabel('Escalation Precision', fontsize=11)
    ax.set_title('Analyst Handoff Quality')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_ablation_results(ablation_df, save_path: str = None):
    """Plot ablation study results"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = ablation_df['variant'].values
    recalls = ablation_df['recall'].values
    ci_lower = ablation_df['ci_lower'].values
    ci_upper = ablation_df['ci_upper'].values
    
    errors = [recalls - ci_lower, ci_upper - recalls]
    
    ax.bar(range(len(variants)), recalls, yerr=errors, capsize=5, color='steelblue')
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Ablation Study: Component Contribution', fontsize=14)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()