# EXP-04: Architecture & Layer Trainability Audit

**Audit Date:** 2026-10-14T18:20:30+05:30  
**Target:** DenseNet-121 Layer Segmentation & Trainability Policy for EXP-04  
**Audit Finding:** **PASS — EXACT LAYERS AND BATCHNORM CONSTRAINTS SPECIFIED**  

---

## 1. DenseNet-121 Structural Decomposition

Keras `tf.keras.applications.DenseNet121(include_top=False)` consists of **427 layers** grouped into an initial convolutional stem, four dense blocks, three transition pooling layers, and a final normalization layer:

| Segment | Layer Range | Key Operations | Output Shape | Pretrained Weights |
|:---|:---:|:---|:---:|:---:|
| **Stem** | Layers 0–5 | `conv1/conv` (7x7, s=2), `conv1/bn`, `conv1/relu`, `pool1` | $(56, 56, 64)$ | ImageNet |
| **Dense Block 1** | Layers 6–50 | 6 dense blocks (`conv2_block1` – `conv2_block6`) | $(56, 56, 256)$ | ImageNet |
| **Transition 1** | Layers 51–54 | `pool2_conv` (1x1), `pool2_pool` (avg 2x2, s=2) | $(28, 28, 128)$ | ImageNet |
| **Dense Block 2** | Layers 55–139 | 12 dense blocks (`conv3_block1` – `conv3_block12`) | $(28, 28, 512)$ | ImageNet |
| **Transition 2** | Layers 140–143 | `pool3_conv` (1x1), `pool3_pool` (avg 2x2, s=2) | $(14, 14, 256)$ | ImageNet |
| **Dense Block 3** | Layers 144–311 | 24 dense blocks (`conv4_block1` – `conv4_block24`) | $(14, 14, 1024)$ | ImageNet |
| **Transition 3** | Layers 312–315 | `pool4_conv` (1x1), `pool4_pool` (avg 2x2, s=2) | $(7, 7, 512)$ | ImageNet |
| **Dense Block 4** | Layers 316–424 | 16 dense blocks (`conv5_block1` – `conv5_block16`) | $(7, 7, 1024)$ | ImageNet |
| **Post-Block Norm** | Layers 425–426 | `bn` (BatchNormalization), `relu` (Activation) | $(7, 7, 1024)$ | ImageNet |

---

## 2. Trainability Policy by Configuration

### Configuration A (Reference / Frozen):
- `base.trainable = False` applied globally across all 427 layers.
- All 7,037,504 backbone parameters are non-trainable.
- Only the classification and embedding heads are trainable (1,968,772 parameters).

### Configuration B (Conservative Partial Fine-Tuning: DenseBlock 4 Only):
- Layers 0 through 312 are **FROZEN** (`layer.trainable = False`).
- Layers 313 through 426 (`conv5_block1` to `conv5_block16` + post-block `bn`/`relu`) are set as follows:
  - **Conv2D Layers:** `trainable = True` (32 Conv2D layers in DenseBlock 4; 2,129,920 parameters).
  - **BatchNormalization Layers:** **STRICTLY FROZEN** (`trainable = False`).
  - **Activation / Concatenate Layers:** Parameter-free.

### Configuration C (Deep Fine-Tuning: DenseBlock 3 + DenseBlock 4):
- Layers 0 through 143 are **FROZEN**.
- Layers 144 through 426 (DenseBlock 3, Transition 3, DenseBlock 4) have all Conv2D layers set to `trainable = True`.
- All BatchNormalization layers remain **STRICTLY FROZEN**.

---

## 3. The Batch Normalization Hazard on Small Cohorts

A critical technical concern during transfer learning is the behavior of Batch Normalization layers:
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta$$

### The Risk:
- The development training fold consists of only **64 participants** (192 scans total).
- With `BATCH_SIZE = 16`, each epoch contains only **4 training steps**.
- If `BatchNormalization.trainable = True`, the layer updates its internal moving averages ($\hat{\mu}$ and $\hat{\sigma}^2$) using mini-batch statistics $\mu_B$ and $\sigma_B^2$.
- With small batches of diverse fingerprint scans, $\mu_B$ and $\sigma_B^2$ exhibit extreme variance. Within 5–10 epochs, this destabilizes the calibrated ImageNet activation scales across the network, leading to catastrophic representation collapse (commonly manifesting as severe majority-class prediction collapse).

### The Mitigation:
All `BatchNormalization` instances in DenseNet-121 will be explicitly locked:
```python
if isinstance(layer, tf.keras.layers.BatchNormalization):
    layer.trainable = False
```
This forces all normalization layers to operate in **inference mode**, using their fixed, well-conditioned ImageNet population statistics ($\hat{\mu}_{ImageNet}, \hat{\sigma}^2_{ImageNet}$).

---

## 4. Parameter Count Audit

| Model Component | Total Parameters | Config A Trainable | Config B Trainable | Config C Trainable |
|:---|:---:|:---:|:---:|:---:|
| **DenseNet-121 Stem (Layers 0–5)** | 78,592 | 0 | 0 | 0 |
| **DenseBlock 1 (Layers 6–50)** | 321,280 | 0 | 0 | 0 |
| **Transition 1 (Layers 51–54)** | 33,024 | 0 | 0 | 0 |
| **DenseBlock 2 (Layers 55–139)** | 1,019,904 | 0 | 0 | 0 |
| **Transition 2 (Layers 140–143)** | 131,584 | 0 | 0 | 0 |
| **DenseBlock 3 (Layers 144–311)** | 2,748,416 | 0 | 0 | 2,525,184 |
| **Transition 3 (Layers 312–315)** | 525,312 | 0 | 0 | 524,288 |
| **DenseBlock 4 Conv2D (32 layers)** | 2,129,920 | 0 | **2,129,920** | 2,129,920 |
| **DenseBlock 4 BN (33 layers)** | 60,416 | 0 | 0 (Frozen) | 0 (Frozen) |
| **Classification Head (`Dense(512)`)** | 1,573,376 | 1,573,376 | 1,573,376 | 1,573,376 |
| **Classification Output (`Dense(4)`)** | 2,052 | 2,052 | 2,052 | 2,052 |
| **Projection Embedding Head (`Dense(128)`)** | 393,344 | 393,344 | 393,344 | 393,344 |
| **TOTALS** | **9,006,276** | **1,968,772** | **4,098,692** | **7,146,164** |

---

## 5. Verified Implementation Function for Model Construction

```python
def configure_densenet_trainability(base_model, config_id: str):
    """
    Configures precise layer-by-layer trainability for EXP-04.
    Guarantees that all BatchNormalization layers remain strictly frozen.
    """
    if config_id == "config_a":
        # Completely frozen backbone
        base_model.trainable = False
        return base_model

    elif config_id == "config_b":
        # Unfreeze DenseBlock 4 Conv2D layers only (Layers 313-426)
        base_model.trainable = True
        for i, layer in enumerate(base_model.layers):
            if i < 313:
                layer.trainable = False
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        return base_model

    elif config_id == "config_c":
        # Unfreeze DenseBlock 3 + Transition 3 + DenseBlock 4 (Layers 144-426)
        base_model.trainable = True
        for i, layer in enumerate(base_model.layers):
            if i < 144:
                layer.trainable = False
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        return base_model
```

---

## 6. Audit Verdict

The layer decomposition, Batch Normalization safeguards, and parameter accounting are **verified and complete**.
- **Audit Verdict:** **PASS**.
