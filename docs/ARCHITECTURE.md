# Signal Encoding & Model Architecture

> **Technical documentation for the Replication Analyzer deep learning system**

This document provides a detailed description of the multi-channel signal encoding strategy and the EXPERT MODEL architecture used for ORI and replication fork detection.

---

## Table of Contents

1. [Signal Encoding Strategy](#signal-encoding-strategy)
2. [Model Architecture](#model-architecture)
3. [Loss Functions](#loss-functions)
4. [Design Rationale](#design-rationale)

---

## Signal Encoding Strategy

The model uses **multi-channel encoding** to represent the raw XY signal data with rich feature representations that capture different aspects of the replication signature.

### Input Data Format

Raw data consists of XY plot files with BrdU/EdU incorporation signal:
- **X-axis**: Position along the DNA read (base pairs)
- **Y-axis**: Signal intensity (normalized analog signal)

Each read is divided into **segments** (windows) for sequence modeling.

### 6-Channel Basic Encoding

The basic encoding creates 6 channels per segment:

```python
def encode_signal_multichannel(signal, smooth_sigma=2, window=50):
    """
    Basic 6-channel encoding

    Channels:
    0. Normalized signal (z-score)
    1. Smoothed signal (Gaussian filter)
    2. First derivative (gradient)
    3. Second derivative (curvature)
    4. Local mean (running average)
    5. Local standard deviation (local variability)
    """
```

#### Channel Descriptions

| Channel | Formula | Purpose |
|---------|---------|---------|
| **0. Normalized** | `z = (x - μ) / σ` | Standardize signal across reads |
| **1. Smoothed** | `s = Gaussian(z, σ=2)` | Denoise while preserving peaks |
| **2. Gradient** | `g = ∇s` | Detect rising/falling edges |
| **3. 2nd Derivative** | `g₂ = ∇²s` | Find inflection points (peak boundaries) |
| **4. Local Mean** | `μ_local = mean(x, window=50)` | Background signal level |
| **5. Local Std** | `σ_local = std(x, window=50)` | Signal variability/noise |

### 9-Channel Enhanced Encoding ⭐

The enhanced encoding adds 3 additional channels for richer representations:

```python
def encode_signal_multichannel_enhanced(signal, smooth_sigma=2, window=50):
    """
    Enhanced 9-channel encoding

    Additional channels:
    6. Z-score (standardized deviation from local mean)
    7. Cumulative sum (trend detection)
    8. Signal envelope (peak boundaries)
    """
```

#### Additional Channels

| Channel | Formula | Purpose |
|---------|---------|---------|
| **6. Z-score** | `z_local = (s - μ_local) / σ_local` | Relative signal strength vs background |
| **7. Cumulative Sum** | `cum = cumsum(s - μ_local)` | Detect sustained trends (drift) |
| **8. Envelope** | `env = max(rolling_max, abs(rolling_min))` | Peak boundaries and amplitude |

### Why Multi-Channel Encoding?

**Problem**: Raw signal alone is noisy and ambiguous
- High background in some regions
- Variable peak shapes
- Different read lengths and qualities

**Solution**: Multi-channel encoding provides:
1. **Noise robustness** (smoothing + local normalization)
2. **Multi-scale features** (raw, smooth, derivatives)
3. **Context awareness** (local statistics)
4. **Peak characterization** (gradients, envelope, z-scores)

### Encoding Pipeline

```
Raw Signal (length L)
    ↓
[Normalize] → Channel 0: Normalized
    ↓
[Smooth with Gaussian σ=2] → Channel 1: Smoothed
    ↓
[Compute ∇] → Channel 2: Gradient
    ↓
[Compute ∇²] → Channel 3: 2nd Derivative
    ↓
[Local statistics (window=50)] → Channels 4-5: Local mean/std
    ↓
[Z-score transform] → Channel 6: Local z-score
    ↓
[Cumulative sum] → Channel 7: Trend
    ↓
[Signal envelope] → Channel 8: Envelope
    ↓
Output: (L, 9) array
```

---

## Model Architecture

The **EXPERT MODEL** uses a sophisticated encoder-decoder architecture combining multiple deep learning components optimized for sequence modeling and peak detection.

### Architecture Overview

```
Input (L × 9)
    ↓
┌─────────────────────────────────┐
│   Multi-Scale CNN               │
│   (Parallel Dilated Convs)      │
│   - Dilation 1, 2, 4            │
└─────────────────────────────────┘
    ↓ (L × 192)
┌─────────────────────────────────┐
│   Encoder                       │
│   - Conv1D(128) + MaxPool       │
│   - Conv1D(256) + MaxPool       │
└─────────────────────────────────┘
    ↓ (L/4 × 256)
┌─────────────────────────────────┐
│   Bidirectional LSTM(128)       │
│   - Forward + Backward context  │
└─────────────────────────────────┘
    ↓ (L/4 × 256)
┌─────────────────────────────────┐
│   Self-Attention                │
│   - Query-Key-Value attention   │
│   - Residual + LayerNorm        │
└─────────────────────────────────┘
    ↓ (L/4 × 256)
┌─────────────────────────────────┐
│   Decoder                       │
│   - UpSample + Conv1D(256)      │
│   - UpSample + Conv1D(128)      │
│   - Crop to original length     │
└─────────────────────────────────┘
    ↓ (L × 128)
┌─────────────────────────────────┐
│   Output Head                   │
│   - Conv1D(64, 3)               │
│   - Conv1D(1, 1, sigmoid)       │
└─────────────────────────────────┘
    ↓
Prediction (L × 1)
```

### Component Details

#### 1. Multi-Scale CNN (Input Layer)

**Purpose**: Capture features at multiple receptive field sizes simultaneously

```python
# Three parallel branches with different dilations
branch1 = Conv1D(64, 7, dilation_rate=1)(input)  # Fine-scale (local)
branch2 = Conv1D(64, 7, dilation_rate=2)(input)  # Medium-scale
branch3 = Conv1D(64, 7, dilation_rate=4)(input)  # Coarse-scale (global)

# Concatenate all branches
multi_scale = Concatenate()([branch1, branch2, branch3])  # → 192 channels
```

**Why dilated convolutions?**
- **Dilation=1**: Captures sharp, narrow peaks (small ORIs, tight forks)
- **Dilation=2**: Captures medium-width features (typical ORIs)
- **Dilation=4**: Captures broad regions and context

**Advantage**: Large receptive field without pooling, preserving resolution

#### 2. Encoder (Downsampling Path)

**Purpose**: Extract hierarchical features while reducing sequence length

```python
# First encoder block
x = Conv1D(128, 5, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)  # L → L/2

# Second encoder block
x = Conv1D(256, 3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)  # L/2 → L/4
```

**Effect**:
- Reduces computational cost for LSTM
- Forces model to learn compressed representations
- Creates feature hierarchy (low-level → high-level)

#### 3. Bidirectional LSTM

**Purpose**: Model long-range temporal dependencies in both directions

```python
lstm_out = Bidirectional(LSTM(128, return_sequences=True))(x)
# Output: L/4 × 256 (128 forward + 128 backward)
```

**Why bidirectional?**
- **Forward pass**: Context from upstream (left side of peak)
- **Backward pass**: Context from downstream (right side of peak)
- Essential for detecting peak boundaries and distinguishing left/right forks

**Why LSTM over GRU?**
- Better gradient flow for long sequences
- Separate forget/input gates provide finer control
- More parameters for complex pattern recognition

#### 4. Self-Attention Layer ⭐

**Purpose**: Attend to important positions regardless of distance

```python
class SelfAttention(layers.Layer):
    def call(self, inputs):
        # Query, Key, Value projections
        Q = Dense(d_model)(inputs)  # Query: "What am I looking for?"
        K = Dense(d_model)(inputs)  # Key: "What information do I have?"
        V = Dense(d_model)(inputs)  # Value: "What is the actual content?"

        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True) / sqrt(d_model)
        attention_weights = softmax(scores)

        # Weighted sum
        attended = tf.matmul(attention_weights, V)

        # Residual connection + LayerNorm
        output = LayerNormalization()(inputs + attended)
        return output
```

**Attention mechanism benefits**:
- Captures **long-range dependencies** (e.g., fork pairs separated by large distances)
- **Selective focus** on important regions (peak centers)
- **Parallel computation** (vs sequential LSTM)

**Mathematical formulation**:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

where:
  Q = inputs × W_Q  (query matrix)
  K = inputs × W_K  (key matrix)
  V = inputs × W_V  (value matrix)
  d_k = dimension of keys (scaling factor)
```

#### 5. Decoder (Upsampling Path)

**Purpose**: Restore original sequence length with learned features

```python
# First decoder block
x = UpSampling1D(2)(x)  # L/4 → L/2
x = Conv1D(256, 3, activation='relu', padding='same')(x)
x = Dropout(0.3)(x)

# Second decoder block
x = UpSampling1D(2)(x)  # L/2 → L
x = Conv1D(128, 3, activation='relu', padding='same')(x)
x = Dropout(0.3)(x)

# Crop to exact original length (handles odd lengths)
x = Cropping1D(cropping=calculate_crop_size())(x)
```

**Upsampling strategy**:
- Simple repetition (UpSampling1D) followed by convolution
- **Alternative**: Transpose convolution (not used due to checkerboard artifacts)

#### 6. Output Head

**Purpose**: Generate per-position predictions

```python
# Feature refinement
x = Conv1D(64, 3, activation='relu', padding='same')(x)
x = Dropout(0.3)(x)

# Final prediction
output = Conv1D(1, 1, activation='sigmoid', padding='same')(x)
```

**For ORI detection (binary)**:
- Output shape: `(L, 1)`
- Activation: `sigmoid` → probability in [0, 1]

**For Fork detection (3-class)**:
- Output shape: `(L, 3)`
- Activation: `softmax` → probability distribution over {background, left, right}

---

## Loss Functions

### Focal Loss (Binary ORI Detection)

**Standard cross-entropy problem**: Overwhelmed by easy negative examples

**Focal Loss solution**: Down-weight well-classified examples

```python
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Focal Loss: FL(p_t) = -α(1 - p_t)^γ log(p_t)

        Parameters:
          alpha: Weight for positive class (0.25 = less weight)
          gamma: Focusing parameter (2.0 = strong focusing)
        """
```

**Mathematical formulation**:

For binary classification:
```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

where:
  p_t = { p     if y = 1 (ORI)
        { 1-p   if y = 0 (background)

  α_t = { α     if y = 1
        { 1-α   if y = 0
```

**Effect of gamma**:
- `γ = 0`: Standard cross-entropy
- `γ = 1`: Moderate focusing
- `γ = 2`: Strong focusing (default) - well-classified examples (p > 0.9) contribute ~100× less loss

**Why focal loss?**
- ORI/fork segments are **rare** (~1-5% of total sequence)
- Standard loss dominated by easy background predictions
- Focal loss forces model to focus on hard examples (peak boundaries, ambiguous regions)

### Multi-Class Focal Loss (Fork Detection)

Extension to 3-class problem: {background, left fork, right fork}

```python
class MultiClassFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=[1.0, 2.0, 2.0], gamma=2.0):
        """
        alpha: Per-class weights [background, left, right]
               Higher weight = more attention to that class
        """
```

**Class weights rationale**:
- `alpha[0] = 1.0`: Background (majority class) - normal weight
- `alpha[1] = 2.0`: Left forks (minority) - double weight
- `alpha[2] = 2.0`: Right forks (minority) - double weight

---

## Design Rationale

### Why This Architecture?

#### 1. **Multi-Scale CNN** instead of single kernel
- ORIs/forks have **variable sizes** (100-2000 bp)
- Single kernel cannot capture all scales
- Parallel dilated convs = efficient multi-scale feature extraction

#### 2. **LSTM** for temporal modeling
- DNA reads are **sequential** data
- Signal patterns depend on context (what came before/after)
- LSTM captures long-range dependencies (e.g., peak shape evolution)

#### 3. **Bidirectional** instead of unidirectional
- Need context from **both directions** to:
  - Locate peak centers precisely
  - Distinguish left vs right forks
  - Define peak boundaries

#### 4. **Self-Attention** for global context
- Some patterns require **very long context** (>1000 bp)
- LSTM struggles with extremely long dependencies
- Attention provides direct connections across entire sequence

#### 5. **Encoder-Decoder** instead of direct CNN
- Downsampling reduces computation (important for long reads)
- Compression forces meaningful representations
- Upsampling restores full resolution for precise peak calling

#### 6. **Focal Loss** instead of cross-entropy
- Extreme class imbalance (~95% background)
- Standard loss produces models that predict "all background"
- Focal loss ensures minority class (ORIs/forks) are learned

### Hyperparameter Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **CNN filters** | 64 | Sufficient capacity without overfitting |
| **LSTM units** | 128 | Good balance (256 bi-directional output) |
| **Dropout** | 0.3 | Prevents overfitting on limited data |
| **Batch size** | 32 | Fits in memory, stable gradients |
| **Learning rate** | 0.0005 | Conservative, stable convergence |
| **Focal α** | 0.25 | Standard for detection tasks |
| **Focal γ** | 2.0 | Strong focusing on hard examples |
| **Dilation rates** | 1,2,4 | Cover range 7bp → 28bp receptive field |

### Alternatives Considered

| Approach | Why Not Used |
|----------|--------------|
| **Transformer** | Too memory-intensive for long sequences (L > 500) |
| **1D U-Net** | Less effective at capturing temporal dependencies |
| **Simple CNN** | Cannot model long-range context needed for peaks |
| **Weighted loss** | Less effective than focal loss for extreme imbalance |
| **Class weights only** | Doesn't address easy vs hard examples |

---

## Performance Characteristics

### Computational Complexity

| Component | Parameters | FLOPs (per sequence) |
|-----------|------------|---------------------|
| Multi-scale CNN | ~120K | O(L) |
| Encoder | ~400K | O(L) |
| BiLSTM | ~530K | O(L) (sequential) |
| Self-Attention | ~260K | O(L²) (attention matrix) |
| Decoder | ~400K | O(L) |
| **Total** | **~1.7M** | **Dominated by O(L²) attention** |

### Memory Requirements

- **Training**: ~2-4 GB RAM (CPU mode, batch_size=32)
- **Inference**: ~500 MB RAM per read
- **Model size**: ~7 MB (.keras file)

### Inference Speed

- **CPU**: ~0.1-0.5 seconds per read (depends on length)
- **GPU**: ~0.01-0.05 seconds per read (if enabled)

---

## Model Variants

### ORI Expert Model (Binary Classification)

```python
model = build_ori_expert_model(max_length=200, n_channels=9)
# Output: (batch, length, 1) with sigmoid activation
```

### Fork Detection Model (Multi-class)

```python
model = build_fork_detection_model(max_length=200, n_channels=9, n_classes=3)
# Output: (batch, length, 3) with softmax activation
# Class 0: background
# Class 1: left fork
# Class 2: right fork
```

**Shared architecture**, only difference is output layer!

---

## Training Strategy

### 1. Data Preparation

```
Raw BED annotations + XY signals
    ↓
[9-channel encoding] → Feature-rich representation
    ↓
[Hybrid balancing] → 50% oversample minority + 50% undersample majority
    ↓
[Sequence padding] → Fixed length (percentile-based)
    ↓
Training data (balanced, padded)
```

### 2. Training Loop

```python
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss=FocalLoss(alpha=0.25, gamma=2.0),
    metrics=[F1Score(), Precision(), Recall(), AUC()]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=[
        EarlyStopping(patience=25, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=10),
        ModelCheckpoint(save_best_only=True)
    ]
)
```

### 3. Evaluation

- **Overall metrics**: Accuracy, F1, Precision, Recall, ROC-AUC
- **Regional metrics**: Separate evaluation for centromeres, pericentromeres, arms
- **Peak calling**: Convert probabilities → discrete regions with thresholding

---

## References

### Key Papers

1. **Focal Loss**: Lin et al. (2017) - "Focal Loss for Dense Object Detection"
2. **Attention Mechanism**: Vaswani et al. (2017) - "Attention Is All You Need"
3. **Bidirectional LSTM**: Graves & Schmidhuber (2005) - "Framewise phoneme classification with bidirectional LSTM"
4. **Dilated Convolutions**: Yu & Koltun (2016) - "Multi-Scale Context Aggregation by Dilated Convolutions"

### Implementation Details

- **Framework**: TensorFlow 2.x / Keras
- **Backend**: CPU-only mode (for reproducibility)
- **Precision**: float32
- **Initialization**: Glorot uniform (default Keras)
- **Regularization**: Dropout (0.3) + BatchNorm
- **Gradient clipping**: Not used (stable with Adam + focal loss)

---

## Summary

The Replication Analyzer architecture combines:

1. **Rich feature encoding** (9 channels) → Robust signal representation
2. **Multi-scale CNN** → Capture variable-size features
3. **Encoder-decoder** → Efficient hierarchical learning
4. **BiLSTM** → Bidirectional temporal context
5. **Self-Attention** → Long-range dependencies
6. **Focal Loss** → Handle extreme class imbalance

This design enables **accurate, robust detection** of replication origins and forks in noisy, variable-length DNA sequencing data.

---

**Document Version**: 1.0
**Last Updated**: 2026-01-03
**Author**: Replication Analyzer Development Team
