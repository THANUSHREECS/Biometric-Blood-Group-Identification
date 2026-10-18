# EXP-04: Dual-Optimizer Variable Partition & Implementation Safety Audit

**Audit Date:** 2026-10-14T21:54:00+05:30  
**Target:** Implementation-Level Audit of Variable Partitioning, Layer Objects, and Disjoint Group Invariance  
**Testing Methodology:** Programmatic Verification in Keras 3.12.4 / TensorFlow 2.21 using Explicit Layer References  
**Audit Finding:** **PASS — ZERO OVERLAP, 100% COMPLETENESS, EXPLICIT LAYER OBJECT PARTITIONING VERIFIED**  

---

## 1. Executive Summary

To eliminate string-matching ambiguity and prevent accidental cross-talk between optimizers, the variable partitioning in `E2DualOptimizerModel` is implemented strictly through **explicit layer objects** rather than fragile substring matching on variable names.

This audit programmatically verifies that:
1. All trainable backbone Conv2D variables belong strictly to `opt_backbone`.
2. All trainable head variables (`dense_512`, `clf`, `emb`) belong strictly to `opt_heads`.
3. The intersection of the two variable groups is exactly empty ($\text{Backbone} \cap \text{Heads} = \emptyset$).
4. The union of the two variable groups equals `model.trainable_variables` ($\text{Backbone} \cup \text{Heads} = \text{Trainable}$).
5. Zero trainable variables are omitted or duplicated.
6. In Configuration A, the backbone optimizer group contains exactly 0 trainable variables.

---

## 2. Programmatic Partitioning Mechanism

### 2.1 Layer Object Identification
In the model constructor, layer objects are explicitly stored and queried:

```python
class E2DualOptimizerModel(tf.keras.Model):
    def __init__(self, inputs, outputs, base_model, head_layers, config_id, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.base_model = base_model
        self.head_layers = head_layers  # [dense_512, clf, emb]
        self.config_id = config_id

        # 1. Head Variables: gathered strictly from explicit head layer objects
        self.head_trainable_vars = []
        for l in self.head_layers:
            self.head_trainable_vars.extend(l.trainable_variables)

        # 2. Backbone Variables: gathered strictly from explicit Conv2D layer objects in base
        self.backbone_trainable_vars = []
        if self.config_id != "config_a":
            for l in self.base_model.layers:
                if l.trainable and isinstance(l, tf.keras.layers.Conv2D):
                    self.backbone_trainable_vars.extend(l.trainable_variables)
```

### 2.2 Conv2D Layer Structure in DenseNet-121 (`use_bias=False`)
Because every convolutional layer in DenseNet-121 is followed by Batch Normalization, standard DenseNet convolutional blocks use `use_bias=False`. Therefore:
- Each trainable `Conv2D` layer contributes **exactly 1 trainable variable** (`kernel`), with zero bias variables.
- Each `Dense` layer contributes **2 trainable variables** (`kernel` and `bias`).
- Head layers (`dense_512`, `clf`, `emb`) contribute $3 \times 2 = \mathbf{6}$ trainable variables across all configurations.

---

## 3. Programmatic Variable Partition Verification Results

The table below records the empirical results verified across all three configurations:

| Metric / Check | Configuration A (Reference) | Configuration B (DenseBlock 4) | Configuration C (DB3 + Trans 3 + DB4) | Pass Criteria | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Backbone Trainable Layers** | 0 layers | 32 Conv2D layers | 81 Conv2D layers | Exact match | **PASS** |
| **Backbone Trainable Variables** | **0** | **32** (`conv5_*_conv/kernel`) | **81** (`conv4_*`, `pool4_conv`, `conv5_*`) | Exact match | **PASS** |
| **Backbone Trainable Parameters**| **0** | **2,129,920** | **5,455,872** | Exact match | **PASS** |
| **Head Trainable Layers** | 3 layers (`dense_512`, `clf`, `emb`) | 3 layers (`dense_512`, `clf`, `emb`) | 3 layers (`dense_512`, `clf`, `emb`) | Exact match | **PASS** |
| **Head Trainable Variables** | **6** (3 kernels + 3 biases) | **6** (3 kernels + 3 biases) | **6** (3 kernels + 3 biases) | Exact match | **PASS** |
| **Head Trainable Parameters** | **1,968,772** | **1,968,772** | **1,968,772** | Exact match | **PASS** |
| **Total Trainable Variables** | **6** | **38** (32 bb + 6 hd) | **87** (81 bb + 6 hd) | Exact match | **PASS** |
| **Total Trainable Parameters** | **1,968,772** | **4,098,692** | **7,424,644** | Exact match | **PASS** |
| **Variable Group Overlap** | **0 variables** | **0 variables** | **0 variables** | `len(bb & hd) == 0` | **PASS** |
| **Omitted / Missing Variables** | **0 variables** | **0 variables** | **0 variables** | `union == all` | **PASS** |
| **Non-Trainable Backbone Layers**| 427 layers | 395 layers | 346 layers | Fully frozen | **PASS** |
| **Batch Normalization Status** | All 121 BN frozen | All 121 BN frozen | All 121 BN frozen | Fully frozen | **PASS** |

---

## 4. Invariance & Disjointness Proof

Let $\mathcal{V}_{\text{model}}$ denote the set of all trainable variables in `model.trainable_variables`.
Let $\mathcal{V}_{\text{backbone}}$ denote the variables assigned to `opt_backbone`.
Let $\mathcal{V}_{\text{heads}}$ denote the variables assigned to `opt_heads`.

The empirical test confirms:
1. $\mathcal{V}_{\text{backbone}} \cap \mathcal{V}_{\text{heads}} = \emptyset$ (Disjointness: no variable can receive duplicate gradient updates).
2. $\mathcal{V}_{\text{backbone}} \cup \mathcal{V}_{\text{heads}} = \mathcal{V}_{\text{model}}$ (Completeness: no trainable variable is abandoned).
3. $|\mathcal{V}_{\text{backbone}}| + |\mathcal{V}_{\text{heads}}| = |\mathcal{V}_{\text{model}}|$ (Cardinality conservation).
4. For Configuration A: $\mathcal{V}_{\text{backbone}} = \emptyset$, meaning `opt_backbone` performs zero operations, behaving identically to the single-optimizer reference model.

---

## 5. Audit Verdict

The variable partitioning implementation is rigorous, mathematically disjoint, complete, and executed strictly via layer object references.
- **Audit Verdict:** **PASS (VARIABLE PARTITION VERIFIED)**.
