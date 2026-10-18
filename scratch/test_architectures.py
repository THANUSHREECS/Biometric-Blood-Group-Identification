import tensorflow as tf
from tensorflow.keras import layers, Model

class GatedScanFusion(layers.Layer):
    """
    Lightweight Gated Feature Fusion across 3 fingerprint scans.
    Computes learnable importance weights alpha_i (sum=1, alpha_i >= 0)
    for each scan and returns F_fused = sum(alpha_i * F_i).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gate_dense = layers.Dense(128, activation="relu", name="gate_dense1")
        self.gate_logit = layers.Dense(1, activation=None, name="gate_logit")
        self.softmax = layers.Softmax(axis=1, name="gate_softmax")

    def call(self, inputs):
        # inputs: list of 3 tensors [f1, f2, f3], each (B, 1024)
        stacked = tf.stack(inputs, axis=1) # (B, 3, 1024)
        h = self.gate_dense(stacked) # (B, 3, 128)
        logits = tf.squeeze(self.gate_logit(h), axis=-1) # (B, 3)
        weights = self.softmax(logits) # (B, 3)
        expanded_weights = tf.expand_dims(weights, axis=-1) # (B, 3, 1)
        fused = tf.reduce_sum(stacked * expanded_weights, axis=1) # (B, 1024)
        return fused

class CrossScanAttentionFusion(layers.Layer):
    """
    Lightweight Cross-Scan Multi-Head Self-Attention across 3 fingerprint scans.
    Maps 3 scans to sequence (B, 3, 256), applies MultiHeadAttention (4 heads, key_dim=64),
    residual connection + LayerNorm, small FFN (512 -> 256), residual + LayerNorm,
    and aggregates via Flatten to (B, 768).
    """
    def __init__(self, num_heads=4, key_dim=64, ffn_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Dense(256, activation=None, name="attn_input_proj")
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name="cross_scan_mha")
        self.ln1 = layers.LayerNormalization(epsilon=1e-5, name="attn_ln1")
        self.ffn1 = layers.Dense(ffn_dim, activation="relu", name="ffn_dense1")
        self.ffn2 = layers.Dense(256, activation=None, name="ffn_dense2")
        self.ln2 = layers.LayerNormalization(epsilon=1e-5, name="attn_ln2")
        self.flatten = layers.Flatten(name="attn_flatten")

    def call(self, inputs):
        # inputs: list of 3 tensors [f1, f2, f3], each (B, 1024)
        stacked = tf.stack(inputs, axis=1) # (B, 3, 1024)
        x = self.proj(stacked) # (B, 3, 256)
        attn_out = self.mha(query=x, key=x, value=x) # (B, 3, 256)
        x = self.ln1(x + attn_out) # (B, 3, 256)
        ffn_out = self.ffn2(self.ffn1(x)) # (B, 3, 256)
        x = self.ln2(x + ffn_out) # (B, 3, 256)
        fused = self.flatten(x) # (B, 768)
        return fused

def build_exp05_model(config_id: str):
    inputs = layers.Input(shape=(3, 112, 112, 3), name="participant_scans")
    s1, s2, s3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]

    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
    base.trainable = False

    pool = layers.GlobalAveragePooling2D(name="gap")
    f1, f2, f3 = pool(base(s1)), pool(base(s2)), pool(base(s3)) # each (B, 1024)

    if config_id == "config_a":
        fused = layers.Concatenate(name="fused_features")([f1, f2, f3]) # (B, 3072)
        dense_512 = layers.Dense(512, activation="relu", name="dense_512")(fused)
        drop = layers.Dropout(0.5, name="dropout_0.5")(dense_512)
        out_clf = layers.Dense(4, activation="softmax", name="clf")(drop)
        out_emb = layers.Dense(128, activation=None, name="emb")(fused)
        model = Model(inputs=inputs, outputs=[out_clf, out_emb], name="E2_EXP05_config_a")

    elif config_id == "config_b":
        fused = GatedScanFusion(name="gated_fusion")([f1, f2, f3]) # (B, 1024)
        dense_512 = layers.Dense(512, activation="relu", name="dense_512")(fused)
        drop = layers.Dropout(0.5, name="dropout_0.5")(dense_512)
        out_clf = layers.Dense(4, activation="softmax", name="clf")(drop)
        out_emb = layers.Dense(128, activation=None, name="emb")(fused)
        model = Model(inputs=inputs, outputs=[out_clf, out_emb], name="E2_EXP05_config_b")

    elif config_id == "config_c":
        fused = CrossScanAttentionFusion(name="attention_fusion")([f1, f2, f3]) # (B, 768)
        dense_512 = layers.Dense(512, activation="relu", name="dense_512")(fused)
        drop = layers.Dropout(0.5, name="dropout_0.5")(dense_512)
        out_clf = layers.Dense(4, activation="softmax", name="clf")(drop)
        out_emb = layers.Dense(128, activation=None, name="emb")(fused)
        model = Model(inputs=inputs, outputs=[out_clf, out_emb], name="E2_EXP05_config_c")

    else:
        raise ValueError(f"Unknown config: {config_id}")

    return model

def main():
    import numpy as np
    x_syn = np.zeros((4, 3, 112, 112, 3), dtype=np.float32)

    for cid in ["config_a", "config_b", "config_c"]:
        m = build_exp05_model(cid)
        preds = m(x_syn, training=False)
        total_p = m.count_params()
        trainable_p = sum(v.numpy().size for v in m.trainable_variables)
        non_trainable_p = sum(v.numpy().size for v in m.non_trainable_variables)
        print(f"\n[{cid.upper()}]")
        print(f"  Total params:         {total_p:,}")
        print(f"  Trainable params:     {trainable_p:,}")
        print(f"  Non-trainable params: {non_trainable_p:,}")
        print(f"  clf output shape:     {preds[0].shape}")
        print(f"  emb output shape:     {preds[1].shape}")
        print(f"  Trainable variables:  {len(m.trainable_variables)}")

if __name__ == "__main__":
    main()
