"""
Visualization Module for Satellite Image Classification.
Generates charts using Matplotlib (headless 'Agg' backend) and returns base64 PNG data.
Supports:
1. Prediction probabilities bar chart
2. Class distribution in dataset
3. Classification history distribution
4. Confusion matrix heatmap
"""

import io
import base64
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web servers
import matplotlib.pyplot as plt
import numpy as np


def fig_to_base64(fig):
    """Encodes a Matplotlib figure into a base64 Data URL string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"


def plot_prediction_probabilities(probabilities_dict):
    """
    Renders a bar chart showing class prediction probabilities.
    Returns base64 data URI string.
    """
    if not probabilities_dict:
        return None

    classes = list(probabilities_dict.keys())
    scores = [float(v) * 100 for v in probabilities_dict.values()]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = ['#2b9348', '#0077b6', '#e76f51', '#6c757d', '#d4a373'][:len(classes)]

    bars = ax.barh(classes, scores, color=colors, height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)', fontsize=10, fontweight='bold')
    ax.set_title('Classification Confidence per Land Class', fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Add numeric labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%", va='center', ha='left', fontsize=9, fontweight='semibold')

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_dataset_distribution(class_counts_dict):
    """
    Renders a bar chart showing count of images per class.
    Returns base64 data URI string.
    """
    if not class_counts_dict or all(v == 0 for v in class_counts_dict.values()):
        return None

    classes = list(class_counts_dict.keys())
    counts = list(class_counts_dict.values())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ['#2a9d8f', '#264653', '#e9c46a', '#f4a261', '#e76f51'][:len(classes)]

    bars = ax.bar(classes, counts, color=colors, width=0.5)
    ax.set_ylabel('Number of Images', fontsize=10, fontweight='bold')
    ax.set_title('Dataset Samples per Class', fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_confusion_matrix(cm_data, class_names):
    """
    Renders confusion matrix heatmap.
    Returns base64 data URI string.
    """
    if not cm_data or not class_names:
        return None

    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap='Blues', alpha=0.85)

    fig.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='left')
    ax.set_yticklabels(class_names)

    ax.set_xlabel('Predicted Label', fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Evaluation Confusion Matrix', fontweight='bold', pad=20)

    # Annotate numbers in cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            ax.text(j, i, f"{val}", ha='center', va='center',
                    color='black' if val < cm.max() / 2 else 'white',
                    fontweight='bold')

    plt.tight_layout()
    return fig_to_base64(fig)
