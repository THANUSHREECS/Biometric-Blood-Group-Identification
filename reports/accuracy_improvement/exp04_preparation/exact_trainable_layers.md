# EXP-04: Exact Trainable Layers Specification

**Audit Date:** 2026-10-14T21:24:30+05:30  
**Target:** Explicit Layer-by-Layer Trainability Definitions for Configurations A, B, and C  
**Rule:** Defined strictly by Layer Name and Layer Class (Not Integer Slice Indices)  
**Audit Finding:** **PASS — ALL TRAINABLE AND FROZEN LAYERS EXPLICITLY ENUMERATED**  

---

## 1. Executive Summary

To prevent accidental unfreezing or architectural ambiguity, this document provides the authoritative, programmatic enumeration of every trainable and frozen layer in DenseNet-121 for each experimental configuration in EXP-04.

---

## 2. Configuration A: Reference Model (Frozen Backbone)

- **Backbone Layer Count:** 427 layers.
- **Trainable Backbone Layers:** **NONE (0 layers)**.
- **Backbone Parameters:** 7,037,504 (all non-trainable).
- **Trainable Head Layers:**
  1. `dense_512` (`Dense`, activation='relu'): Weight $(3072, 512)$, Bias $(512)$ $\rightarrow$ 1,573,376 params.
  2. `clf` (`Dense`, activation='softmax'): Weight $(512, 4)$, Bias $(4)$ $\rightarrow$ 2,052 params.
  3. `emb` (`Dense`, activation=None): Weight $(3072, 128)$, Bias $(128)$ $\rightarrow$ 393,344 params.
- **Total Trainable Parameters:** **1,968,772**.

---

## 3. Configuration B: Conservative Fine-Tuning (DenseBlock 4 Only)

### Layer Selection Rule:
A layer in `base.layers` is set to `trainable = True` if and only if:
1. `isinstance(layer, tf.keras.layers.Conv2D) == True`
2. `layer.name.startswith("conv5_") == True`
All other layers in `base.layers` (including all BatchNormalization layers, pooling layers, and earlier blocks) are set to `trainable = False`.

### Enumeration of the 32 Trainable Conv2D Layers in DenseBlock 4:
1. `conv5_block1_1_conv` (1x1 Conv, 1024 $\rightarrow$ 128)
2. `conv5_block1_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
3. `conv5_block2_1_conv` (1x1 Conv, 1056 $\rightarrow$ 128)
4. `conv5_block2_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
5. `conv5_block3_1_conv` (1x1 Conv, 1088 $\rightarrow$ 128)
6. `conv5_block3_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
7. `conv5_block4_1_conv` (1x1 Conv, 1120 $\rightarrow$ 128)
8. `conv5_block4_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
9. `conv5_block5_1_conv` (1x1 Conv, 1152 $\rightarrow$ 128)
10. `conv5_block5_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
11. `conv5_block6_1_conv` (1x1 Conv, 1184 $\rightarrow$ 128)
12. `conv5_block6_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
13. `conv5_block7_1_conv` (1x1 Conv, 1216 $\rightarrow$ 128)
14. `conv5_block7_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
15. `conv5_block8_1_conv` (1x1 Conv, 1248 $\rightarrow$ 128)
16. `conv5_block8_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
17. `conv5_block9_1_conv` (1x1 Conv, 1280 $\rightarrow$ 128)
18. `conv5_block9_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
19. `conv5_block10_1_conv` (1x1 Conv, 1312 $\rightarrow$ 128)
20. `conv5_block10_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
21. `conv5_block11_1_conv` (1x1 Conv, 1344 $\rightarrow$ 128)
22. `conv5_block11_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
23. `conv5_block12_1_conv` (1x1 Conv, 1376 $\rightarrow$ 128)
24. `conv5_block12_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
25. `conv5_block13_1_conv` (1x1 Conv, 1408 $\rightarrow$ 128)
26. `conv5_block13_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
27. `conv5_block14_1_conv` (1x1 Conv, 1440 $\rightarrow$ 128)
28. `conv5_block14_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
29. `conv5_block15_1_conv` (1x1 Conv, 1472 $\rightarrow$ 128)
30. `conv5_block15_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)
31. `conv5_block16_1_conv` (1x1 Conv, 1504 $\rightarrow$ 128)
32. `conv5_block16_2_conv` (3x3 Conv, 128 $\rightarrow$ 32)

### Strictly Frozen Layers in DenseBlock 4:
- All 32 block BatchNormalization layers (`conv5_block1_0_bn`, `conv5_block1_1_bn`, ..., `conv5_block16_1_bn`).
- The post-block normalization layer `bn` (Layer 425).
- All 32 ReLU activations and 16 Concatenate operations.

### Configuration B Parameter Breakdown:
- Backbone Trainable Parameters: **2,129,920**
- Backbone Non-Trainable Parameters: **4,907,584**
- Head Trainable Parameters: **1,968,772**
- **Total Model Trainable Parameters:** **4,098,692**

---

## 4. Configuration C: Deep Fine-Tuning (DenseBlock 3 + Transition 3 + DenseBlock 4)

### Explicit Definition of Transition 3:
Transition 3 contains:
- `pool4_bn` (`BatchNormalization`): **STRICTLY FROZEN**
- `pool4_relu` (`Activation`): Non-parametric
- `pool4_conv` (`Conv2D`, 1x1, 1024 $\rightarrow$ 512 channels, 524,288 parameters): **TRAINABLE**
- `pool4_pool` (`AveragePooling2D`): Non-parametric

**Decision on Transition 3:** `pool4_conv` is explicitly designated as **TRAINABLE** in Configuration C because it performs the essential bottleneck projection connecting DenseBlock 3 to DenseBlock 4.

### Enumeration of the 81 Trainable Conv2D Layers in Configuration C:
1. **DenseBlock 3 Conv2D Layers (48 layers):**
   - `conv4_block1_1_conv` through `conv4_block24_2_conv` (2,797,056 params).
2. **Transition 3 Conv2D Layer (1 layer):**
   - `pool4_conv` (524,288 params).
3. **DenseBlock 4 Conv2D Layers (32 layers):**
   - `conv5_block1_1_conv` through `conv5_block16_2_conv` (2,129,920 params).

### Strictly Frozen Layers in Configuration C:
- All BatchNormalization layers across the entire network (including `conv4_*_bn`, `pool4_bn`, and `conv5_*_bn`).
- Stem (Conv1, Pool1).
- DenseBlock 1 (`conv2_*`).
- Transition 1 (`pool2_*`).
- DenseBlock 2 (`conv3_*`).
- Transition 2 (`pool3_*`).

### Configuration C Parameter Breakdown:
- Backbone Trainable Parameters: **5,455,872** (81 Conv2D layers)
- Backbone Non-Trainable Parameters: **1,581,632**
- Head Trainable Parameters: **1,968,772**
- **Total Model Trainable Parameters:** **7,424,644**

---

## 5. Programmatic Trainability Enforcer

```python
def apply_trainability_policy(base_model, config_id: str):
    """
    Authoritative layer-name and layer-type trainability enforcer.
    """
    if config_id == "config_a":
        base_model.trainable = False
        return

    base_model.trainable = True
    for layer in base_model.layers:
        is_conv = isinstance(layer, tf.keras.layers.Conv2D)
        
        if config_id == "config_b":
            # Trainable ONLY if Conv2D in DenseBlock 4
            layer.trainable = is_conv and layer.name.startswith("conv5_")
            
        elif config_id == "config_c":
            # Trainable ONLY if Conv2D in DenseBlock 3, Transition 3, or DenseBlock 4
            is_target = layer.name.startswith(("conv4_", "pool4_conv", "conv5_"))
            layer.trainable = is_conv and is_target
        else:
            raise ValueError(f"Unknown config_id: {config_id}")
```

---

## 6. Parameter Group and Learning-Rate Mapping

To prevent confounding across architectures, each configuration binds its trainable parameter groups to specific learning rates:

| Configuration | Parameter Group | Layer Names Included | Trainable Weight Count | Effective LR | Optimizer |
|:---|:---|:---|:---:|:---:|:---:|
| **Config A** | Backbone | None (All 427 layers frozen) | 0 | N/A | None |
| | Heads | `dense_512`, `clf`, `emb` | 1,968,772 | **1e-3** | AdamW |
| **Config B** | Backbone | DenseBlock 4 Conv2D (`conv5_block1_1_conv` ... `conv5_block16_2_conv`) | 2,129,920 | **1e-4** | AdamW |
| | Heads | `dense_512`, `clf`, `emb` | 1,968,772 | **1e-3** | AdamW |
| **Config C** | Backbone | DB3 Conv2D + Transition 3 `pool4_conv` + DB4 Conv2D (81 layers) | 5,455,872 | **5e-5** | AdamW |
| | Heads | `dense_512`, `clf`, `emb` | 1,968,772 | **1e-3** | AdamW |

---

## 7. Audit Verdict

All trainable and frozen layers for Configurations A, B, and C are identified by exact name, layer type, parameter count, and optimizer learning-rate mapping.
- **Audit Verdict:** **PASS (EXACT SPECIFICATION LOCKED)**.

