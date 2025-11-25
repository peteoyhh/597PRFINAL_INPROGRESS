"""
Plotting utilities for Mahjong Monte-Carlo experiments.

All plots are saved as PNG files (dpi=200) in non-interactive mode.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)


def save_line_plot(x, y, title, xlabel, ylabel, outfile, y2=None, label1=None, label2=None, legend=True):
    """
    Save a line plot.
    
    Args:
        x: X-axis values
        y: Y-axis values (single line)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        outfile: Output file path
        y2: Optional second Y-axis values for dual-line plot
        label1: Optional label for first line (default: ylabel if no y2, else 'Line 1')
        label2: Optional label for second line
        legend: Whether to show legend
    """
    plt.figure(figsize=(8, 6))
    # Use label1 if provided, otherwise use ylabel if no y2, else default to 'Line 1'
    first_label = label1 if label1 is not None else (ylabel if not y2 else 'Line 1')
    plt.plot(x, y, marker='o', linewidth=2, markersize=6, label=first_label)
    
    if y2 is not None:
        plt.plot(x, y2, marker='s', linewidth=2, markersize=6, label=label2 or 'Line 2')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel if y2 is None else 'Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if legend and (y2 is not None or label2 is not None):
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()


def save_bar_plot(labels, values, title, outfile, ylabel="Value", color=None):
    """
    Save a bar chart.
    
    Args:
        labels: X-axis labels
        values: Y-axis values
        title: Plot title
        outfile: Output file path
        ylabel: Y-axis label
        color: Bar color (optional)
    """
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Category', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()


def save_hist(data, title, outfile, xlabel="Value", ylabel="Frequency", bins=20, density=False):
    """
    Save a histogram.
    
    Args:
        data: Data array
        title: Plot title
        outfile: Output file path
        xlabel: X-axis label
        ylabel: Y-axis label
        bins: Number of bins
        density: Whether to normalize as density
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, density=density, alpha=0.7, edgecolor='black', linewidth=1.5)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()


def save_scatter_plot(x, y, title, xlabel, ylabel, outfile, alpha=0.5):
    """
    Save a scatter plot.
    
    Args:
        x: X-axis values
        y: Y-axis values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        outfile: Output file path
        alpha: Transparency (0-1)
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=alpha, s=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()


def save_multi_bar_plot(labels, data_dict, title, outfile, ylabel="Value"):
    """
    Save a grouped bar chart.
    
    Args:
        labels: X-axis labels
        data_dict: Dictionary of {series_name: [values]}
        title: Plot title
        outfile: Output file path
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (series_name, values) in enumerate(data_dict.items()):
        offset = (i - len(data_dict)/2 + 0.5) * width / len(data_dict)
        plt.bar(x + offset, values, width/len(data_dict), label=series_name, 
                color=colors[i % len(colors)], alpha=0.7, edgecolor='black', linewidth=1)
    
    plt.xlabel('Category', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()

