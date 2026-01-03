#!/usr/bin/env python
"""
Generate gallery images for documentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

output_dir = Path(__file__).parent / 'images'
output_dir.mkdir(exist_ok=True)


def generate_architecture_diagram():
    """Generate model architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(5, 11.5, 'EXPERT MODEL Architecture',
            fontsize=20, fontweight='bold', ha='center')
    ax.text(5, 11, 'Multi-scale CNN + BiLSTM + Self-Attention',
            fontsize=12, ha='center', style='italic', color='gray')

    # Colors
    input_color = '#E8F4F8'
    cnn_color = '#B8E6F0'
    lstm_color = '#88D8E8'
    attention_color = '#58CAE0'
    output_color = '#90EE90'

    # Input layer
    y = 10
    box = FancyBboxPatch((2, y), 6, 0.6, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=input_color, linewidth=2)
    ax.add_patch(box)
    ax.text(5, y+0.3, 'Input: 9-channel Signal (L × 9)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Multi-scale CNN branches
    y = 8.5
    ax.text(5, y+1, 'Multi-Scale CNN (Parallel Branches)',
            ha='center', fontsize=12, fontweight='bold')

    branches = [
        (1.5, 'Dilation=1\nConv1D(64,7)', 2),
        (4, 'Dilation=2\nConv1D(64,7)', 2.5),
        (6.5, 'Dilation=4\nConv1D(64,7)', 2)
    ]

    for x, label, width in branches:
        box = FancyBboxPatch((x, y), width, 0.8, boxstyle="round,pad=0.05",
                              edgecolor='blue', facecolor=cnn_color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x+width/2, y+0.4, label, ha='center', va='center',
                fontsize=9, fontweight='bold')

    # Concatenate
    y = 7.3
    box = FancyBboxPatch((3, y), 4, 0.5, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='#D0D0D0', linewidth=1.5)
    ax.add_patch(box)
    ax.text(5, y+0.25, 'Concatenate → 192 channels',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Encoder
    y = 6.2
    ax.text(5, y+0.7, 'Encoder (Downsampling)',
            ha='center', fontsize=11, fontweight='bold')

    box = FancyBboxPatch((2.5, y), 5, 0.5, boxstyle="round,pad=0.05",
                          edgecolor='darkblue', facecolor=cnn_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(5, y+0.25, 'Conv1D(128,5) + MaxPool → L/2',
            ha='center', va='center', fontsize=9)

    y = 5.5
    box = FancyBboxPatch((2.5, y), 5, 0.5, boxstyle="round,pad=0.05",
                          edgecolor='darkblue', facecolor=cnn_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(5, y+0.25, 'Conv1D(256,3) + MaxPool → L/4',
            ha='center', va='center', fontsize=9)

    # BiLSTM
    y = 4.3
    box = FancyBboxPatch((2, y), 6, 0.6, boxstyle="round,pad=0.1",
                          edgecolor='darkgreen', facecolor=lstm_color, linewidth=2)
    ax.add_patch(box)
    ax.text(5, y+0.3, 'Bidirectional LSTM (128 units → 256 channels)',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Self-Attention
    y = 3.3
    box = FancyBboxPatch((1.5, y), 7, 0.6, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor=attention_color, linewidth=2)
    ax.add_patch(box)
    ax.text(5, y+0.3, 'Self-Attention (Query-Key-Value) + Residual + LayerNorm',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Decoder
    y = 2.3
    ax.text(5, y+0.7, 'Decoder (Upsampling)',
            ha='center', fontsize=11, fontweight='bold')

    box = FancyBboxPatch((2.5, y), 5, 0.5, boxstyle="round,pad=0.05",
                          edgecolor='darkblue', facecolor=cnn_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(5, y+0.25, 'UpSample(×2) + Conv1D(256,3) → L/2',
            ha='center', va='center', fontsize=9)

    y = 1.6
    box = FancyBboxPatch((2.5, y), 5, 0.5, boxstyle="round,pad=0.05",
                          edgecolor='darkblue', facecolor=cnn_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(5, y+0.25, 'UpSample(×2) + Conv1D(128,3) + Crop → L',
            ha='center', va='center', fontsize=9)

    # Output
    y = 0.5
    box = FancyBboxPatch((3, y), 4, 0.5, boxstyle="round,pad=0.05",
                          edgecolor='darkgreen', facecolor=output_color, linewidth=2)
    ax.add_patch(box)
    ax.text(5, y+0.25, 'Conv1D(64,3) → Conv1D(1,1,sigmoid)',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Add arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='gray')
    positions_y = [10, 8.5, 7.3, 6.2, 5.5, 4.3, 3.3, 2.3, 1.6, 0.5]
    for i in range(len(positions_y)-1):
        ax.annotate('', xy=(5, positions_y[i+1]+0.5), xytext=(5, positions_y[i]),
                    arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig(output_dir / 'architecture.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'architecture.png'}")
    plt.close()


def generate_training_example():
    """Generate example training history plot."""
    np.random.seed(42)
    epochs = np.arange(1, 151)

    # Simulate realistic training curves
    train_loss = 0.5 * np.exp(-epochs/30) + 0.1 + np.random.normal(0, 0.02, len(epochs))
    val_loss = 0.5 * np.exp(-epochs/35) + 0.12 + np.random.normal(0, 0.03, len(epochs))

    train_f1 = 1 - 0.5 * np.exp(-epochs/25) - 0.08 + np.random.normal(0, 0.015, len(epochs))
    val_f1 = 1 - 0.5 * np.exp(-epochs/30) - 0.10 + np.random.normal(0, 0.025, len(epochs))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, train_loss, label='Training Loss', linewidth=2, color='#3498db')
    axes[0, 0].plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='#e74c3c')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # F1-Score
    axes[0, 1].plot(epochs, train_f1, label='Training F1', linewidth=2, color='#2ecc71')
    axes[0, 1].plot(epochs, val_f1, label='Validation F1', linewidth=2, color='#f39c12')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('F1-Score', fontsize=12)
    axes[0, 1].set_title('F1-Score Progress', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.3, 1.0])

    # Precision vs Recall
    precision = train_f1 + np.random.normal(0, 0.02, len(epochs))
    recall = train_f1 - np.random.normal(0, 0.02, len(epochs))
    axes[1, 0].plot(epochs, precision, label='Precision', linewidth=2, color='#9b59b6')
    axes[1, 0].plot(epochs, recall, label='Recall', linewidth=2, color='#1abc9c')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('Precision & Recall', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0.3, 1.0])

    # Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
    TRAINING SUMMARY
    {'='*40}

    Best Validation Loss: {val_loss.min():.4f} (Epoch {val_loss.argmin()+1})
    Best Validation F1: {val_f1.max():.4f} (Epoch {val_f1.argmax()+1})

    Final Metrics:
      • Train Loss: {train_loss[-1]:.4f}
      • Val Loss: {val_loss[-1]:.4f}
      • Train F1: {train_f1[-1]:.4f}
      • Val F1: {val_f1[-1]:.4f}

    Model: ORI Expert
    Total Epochs: 150
    Batch Size: 32
    Learning Rate: 0.0005
    Loss Function: Focal Loss (α=0.25, γ=2.0)
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11,
                    family='monospace', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Example Training History - ORI Detection Model',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'training_history.png'}")
    plt.close()


def generate_prediction_example():
    """Generate example prediction visualization."""
    np.random.seed(123)

    # Simulate a read with an ORI
    length = 2000
    x = np.arange(length)

    # Base signal with noise
    signal = np.random.normal(0.3, 0.1, length)

    # Add ORI peak (800-1200)
    ori_center = 1000
    ori_width = 200
    for i in range(length):
        if abs(i - ori_center) < ori_width:
            signal[i] += 0.6 * np.exp(-((i-ori_center)**2)/(2*(ori_width/3)**2))

    # Smooth signal
    from scipy.ndimage import gaussian_filter1d
    signal_smooth = gaussian_filter1d(signal, sigma=10)

    # Simulated prediction
    prediction = np.zeros(length)
    for i in range(length):
        if abs(i - ori_center) < ori_width:
            prediction[i] = 0.9 * np.exp(-((i-ori_center)**2)/(2*(ori_width/2.5)**2))
        else:
            prediction[i] = np.random.uniform(0, 0.1)

    prediction = gaussian_filter1d(prediction, sigma=5)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Raw signal
    axes[0].plot(x, signal, alpha=0.5, color='gray', linewidth=0.5, label='Raw')
    axes[0].plot(x, signal_smooth, color='#3498db', linewidth=2, label='Smoothed')
    axes[0].set_ylabel('Signal Intensity', fontsize=12)
    axes[0].set_title('Input Signal (XY Data)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.2, 1.2])

    # Prediction probability
    axes[1].plot(x, prediction, color='#e74c3c', linewidth=2.5, label='ORI Probability')
    axes[1].axhline(0.5, color='black', linestyle='--', linewidth=1.5,
                    alpha=0.5, label='Threshold (0.5)')
    axes[1].fill_between(x, 0, prediction, where=(prediction > 0.5),
                         color='#e74c3c', alpha=0.3, label='Called ORI')
    axes[1].set_ylabel('Probability', fontsize=12)
    axes[1].set_title('Model Prediction', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.05, 1.05])

    # Called regions
    axes[2].fill_between(x, 0, 1, where=(prediction > 0.5),
                         color='#2ecc71', alpha=0.6, label='ORI Region')
    axes[2].set_ylabel('Called Region', fontsize=12)
    axes[2].set_xlabel('Position in Read (bp)', fontsize=12)
    axes[2].set_title('Final Annotation', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 1])
    axes[2].set_yticks([])
    axes[2].grid(True, alpha=0.3, axis='x')

    # Annotate ORI
    axes[2].text(ori_center, 0.5, f'ORI\n{ori_center-ori_width}–{ori_center+ori_width} bp',
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))

    plt.suptitle('Example: ORI Detection on a Single Read',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'example_prediction.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'example_prediction.png'}")
    plt.close()


def generate_workflow_diagram():
    """Generate workflow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Title
    ax.text(5, 10.5, 'Replication Analyzer Workflow',
            fontsize=20, fontweight='bold', ha='center')

    # Colors for different stages
    data_color = '#E8F4F8'
    process_color = '#FFE4B5'
    model_color = '#D8BFD8'
    output_color = '#90EE90'

    y = 9.5

    # Step 1: Data Input
    box = FancyBboxPatch((1, y), 8, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='#2980b9', facecolor=data_color, linewidth=2)
    ax.add_patch(box)
    ax.text(5, y+0.4, '1. Data Input: BED files + XY signal files',
            ha='center', va='center', fontsize=12, fontweight='bold')

    y = 8.3
    # Arrow
    ax.annotate('', xy=(5, y+0.5), xytext=(5, y+0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    # Step 2: Preprocessing
    box = FancyBboxPatch((1, y), 8, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='#d35400', facecolor=process_color, linewidth=2)
    ax.add_patch(box)
    ax.text(5, y+0.4, '2. Preprocessing: 9-channel encoding + Hybrid balancing',
            ha='center', va='center', fontsize=12, fontweight='bold')

    y = 7.1
    ax.annotate('', xy=(5, y+0.5), xytext=(5, y+0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    # Step 3: Training
    box = FancyBboxPatch((1, y), 8, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='#8e44ad', facecolor=model_color, linewidth=2)
    ax.add_patch(box)
    ax.text(5, y+0.4, '3. Training: CNN+BiLSTM+Attention with Focal Loss',
            ha='center', va='center', fontsize=12, fontweight='bold')

    y = 5.9
    ax.annotate('', xy=(5, y+0.5), xytext=(5, y+0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    # Step 4: Evaluation
    box = FancyBboxPatch((1, y), 8, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='#16a085', facecolor=output_color, linewidth=2)
    ax.add_patch(box)
    ax.text(5, y+0.4, '4. Evaluation: Per-region metrics + Comprehensive plots',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # Branch: Two paths
    y = 4.9
    ax.text(5, y, '⬇', ha='center', fontsize=20, color='gray')

    # Left path: Re-train
    y_left = 3.8
    ax.annotate('', xy=(2.5, y_left+0.7), xytext=(4, y-0.3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', linestyle='dashed'))

    box = FancyBboxPatch((0.5, y_left), 4, 0.6, boxstyle="round,pad=0.05",
                          edgecolor='gray', facecolor='#f0f0f0', linewidth=1, linestyle='dashed')
    ax.add_patch(box)
    ax.text(2.5, y_left+0.3, 'Adjust hyperparameters\n& retrain',
            ha='center', va='center', fontsize=10, style='italic')

    # Arrow back up
    ax.annotate('', xy=(1.5, 7.1), xytext=(1.5, y_left),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', linestyle='dashed'))

    # Right path: Deploy
    y_right = 3.8
    ax.annotate('', xy=(7.5, y_right+0.7), xytext=(6, y-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='#27ae60'))

    box = FancyBboxPatch((5.5, y_right), 4, 0.6, boxstyle="round,pad=0.05",
                          edgecolor='#27ae60', facecolor='#d5f4e6', linewidth=2)
    ax.add_patch(box)
    ax.text(7.5, y_right+0.3, 'Model OK ✓\nProceed to annotation',
            ha='center', va='center', fontsize=10, fontweight='bold', color='#27ae60')

    y = 2.8
    ax.annotate('', xy=(7.5, y+0.5), xytext=(7.5, y_right),
                arrowprops=dict(arrowstyle='->', lw=2, color='#27ae60'))

    # Step 5: Annotate New Data
    box = FancyBboxPatch((4.5, y), 6, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='#c0392b', facecolor='#ffeaa7', linewidth=3)
    ax.add_patch(box)
    ax.text(7.5, y+0.4, '5. Annotate New Data 🔬',
            ha='center', va='center', fontsize=13, fontweight='bold')

    y = 1.6
    ax.annotate('', xy=(7.5, y+0.5), xytext=(7.5, y+0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    # Step 6: Output
    box = FancyBboxPatch((4.5, y), 6, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='#27ae60', facecolor=output_color, linewidth=2)
    ax.add_patch(box)
    ax.text(7.5, y+0.4, '6. Outputs: BED files + TSV + Plots',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # Add note boxes
    y = 0.5
    note_text = """
    KEY SCRIPTS:
    • train_ori_model.py / train_fork_model.py
    • evaluate_model.py
    • annotate_new_data.py ⭐
    """
    ax.text(2, y, note_text, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.8, edgecolor='#856404'))

    note_text2 = """
    OUTPUTS:
    • Segment predictions (TSV)
    • Called peaks (BED/TSV)
    • Visualization plots
    """
    ax.text(7, y, note_text2, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#d4edda', alpha=0.8, edgecolor='#155724'))

    plt.tight_layout()
    plt.savefig(output_dir / 'workflow.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'workflow.png'}")
    plt.close()


if __name__ == '__main__':
    print("Generating gallery images...")
    print()

    generate_architecture_diagram()
    generate_training_example()
    generate_prediction_example()
    generate_workflow_diagram()

    print()
    print("✅ All gallery images generated successfully!")
    print(f"   Location: {output_dir.absolute()}")
