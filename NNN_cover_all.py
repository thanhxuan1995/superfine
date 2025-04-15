# %%
# data prepare
import pandas as pd
import numpy as np

# %%
data = pd.read_csv("huge_input_data.csv")

# %%
df = data
df = df.dropna().reset_index(drop=True)
df["revenue"] = df["revenue"].astype(str).str.replace(",", "").astype(float)

# Convert install_date to datetime
df["install_date"] = pd.to_datetime(df["install_date"])

df.sort_values("install_date", inplace=True)
df.reset_index(drop=True, inplace=True)
# Step 1: Create full date range
date_range = pd.date_range(start=df["install_date"].min(), end=df["install_date"].max())
date_idx = {date: i for i, date in enumerate(date_range)}
num_dates = len(date_range)

# %%
# Other index mappings
geo_idx = {geo: i for i, geo in enumerate(dff["country"].dropna().unique())}
channel_idx = {ch: i for i, ch in enumerate(dff["channel_id"].unique())}
platform_idx = {p: i for i, p in enumerate(dff["platform"].unique())}
features = ["spend", "revenue", "clicks", "impressions", "installs"]

# %%
# Step 2: Initialize the tensor
GTCD = np.zeros(
    (len(platform_idx), len(geo_idx), len(channel_idx), num_dates, len(features))
)

# Step 3: Populate tensor
for _, row in dff.iterrows():
    try:
        g = geo_idx[row["country"]]
        c = channel_idx[row["channel_id"]]
        p = platform_idx[row["platform"]]
        d = date_idx[pd.to_datetime(row["install_date"])]
        GTCD[p, g, c, d, :] += [
            pd.to_numeric(row[feat], errors="coerce") for feat in features
        ]
    except KeyError:
        # Skip rows with missing country or unmatched date
        continue

# %%
GTCD.shape

# %%
### scale input data: need to change ()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
flat_GTCD = GTCD.reshape(-1, GTCD.shape[-1])  # (G*T*C, D)
GTCD_scaled = scaler.fit_transform(flat_GTCD).reshape(GTCD.shape)
GTCD_scaled.shape

# %%
# convert tesnfor flow data:
import tensorflow as tf

# Convert to tf.Tensor
GTCD_tensor = tf.convert_to_tensor(GTCD_scaled, dtype=tf.float32)
GTCD_tensor.shape

# %%
# target: shape (G, T, D) — same as input, but as labels
target = GTCD_tensor  # shape: (P, G, C, T, D)
target = tf.reduce_mean(target, axis=[3])  # average over channels → (G, T, D)
target.shape, GTCD_tensor.shape  ## want to return every 87 datse resut with 5 features.

# %%
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LayerNormalization,
    MultiHeadAttention,
    Dropout,
    GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam
import pandas as pd

# %%
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(key_dim),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        attn_output = self.att(x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        return self.norm2(x + ffn_output)

# %%
class NNNTransformer(tf.keras.Model):
    def __init__(
        self, embedding_dim=5, num_heads=1, num_layers=2, ff_dim=64, output_dim=5
    ):
        super().__init__()
        self.proj = tf.keras.layers.Dense(embedding_dim)
        self.transformers = [
            TransformerBlock(num_heads, embedding_dim, ff_dim)
            for _ in range(num_layers)
        ]
        self.output_head = tf.keras.layers.Dense(
            output_dim
        )  # <-- now outputs 5 features

    def call(self, inputs):
        P = tf.shape(inputs)[0]
        G = tf.shape(inputs)[1]
        C = tf.shape(inputs)[2]
        T = tf.shape(inputs)[3]
        D = tf.shape(inputs)[4]
        print("Printing P, G, C, T, D :", P, G, C, T, D)
        # Average across P, G, C to get (T, D)
        x = tf.reshape(inputs, (-1, T, D))  # (P*G*C, T, D)
        x = self.proj(x)
        print("Shape after projection:", x.shape)

        for layer in self.transformers:
            x = layer(x)
            print("Shape after transformer layer:", x.shape)

        x = tf.reduce_mean(x, axis=1)  # (P*G*C, D)
        print("Shape after xuan reduce transformer layer:", x.shape)
        x = self.output_head(x)  # (P*G*C, 5)
        print("Shape after output hed:", x.shape)
        x = tf.reshape(x, (P, G, C, 5))  # (P, G, C, 5)
        print("Shape at final output", x.shape)
        return x

# %%
target.shape, GTCD_tensor.shape

# %%
# X_train = tf.convert_to_tensor(GTCD_tensor, dtype=tf.float32)  # shape: (2, 172, 6, 87, 5)
# y_train = tf.convert_to_tensor(target, dtype=tf.float32)

# %%
print("Any NaN in inputs?", tf.math.reduce_any(tf.math.is_nan(GTCD_tensor)).numpy())
print("Any Inf in inputs?", tf.math.reduce_any(tf.math.is_inf(GTCD_tensor)).numpy())
print("Any NaN in targets?", tf.math.reduce_any(tf.math.is_nan(target)).numpy())
print("Any Inf in targets?", tf.math.reduce_any(tf.math.is_inf(target)).numpy())

# %%
GTCD_tensor = tf.where(
    tf.math.is_finite(GTCD_tensor), GTCD_tensor, tf.zeros_like(GTCD_tensor)
)
target = tf.where(tf.math.is_finite(target), target, tf.zeros_like(target))

# %%
dataset = tf.data.Dataset.from_tensor_slices((GTCD_tensor, target)).batch(2)

# %%
dataset = (
    tf.data.Dataset.from_tensor_slices((GTCD_tensor, target))
    .shuffle(buffer_size=10)
    .batch(2)
    .prefetch(tf.data.AUTOTUNE)
)

# %%
model = NNNTransformer(output_dim=5, num_heads=2, num_layers=2, ff_dim=64)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(dataset, epochs=10)

# %%
preds = model.predict(GTCD_tensor)  # shape: (P, G, C, 5)

# %%
preds

# %%
import tensorflow as tf

# Instantiate the model
model = NNNTransformer(
    embedding_dim=16, num_heads=2, num_layers=2, ff_dim=64, output_dim=5
)
# Example input (2, 172, 6, 87, 5)
GTCD_tensor = tf.random.normal((2, 172, 6, 87, 5))
# Make prediction
predictions = model(GTCD_tensor, training=False)
print("Predictions shape:", predictions.shape)

# %%
columns = ["spend", "revenue", "clicks", "impressions", "installs"]
results = []
# Other index mappings
# geo_idx = {geo: i for i, geo in enumerate(dff["country"].dropna().unique())}
# channel_idx = {ch: i for i, ch in enumerate(dff["channel_id"].unique())}
# platform_idx = {p: i for i, p in enumerate(dff["platform"].unique())}
# features = ["spend", "revenue", "clicks", "impressions", "installs"]
# (P, G, C, 5)
for p_idx, plat in enumerate(platform_idx):
    for g_idx, geo in enumerate(geo_idx):
        for c_idx, channel in enumerate(channel_idx):
            row = preds[p_idx, g_idx, c_idx]
            results.append(
                {
                    "platform": plat,
                    "country": geo,
                    "channel": channel,
                    **{col: row[i] for i, col in enumerate(columns)},
                }
            )

pred_df = pd.DataFrame(results)

# %%
pred_df

# %%



