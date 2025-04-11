# %%
import numpy as np

# Shape: (samples, time_steps, features)
# Each feature could represent spend for one channel
X = np.random.rand(500, 30, 5)  # 500 samples, 30 time steps, 5 channels
y = np.random.rand(500, 1)  # Total conversions or revenue per sample

# %%


# %%
import pandas as pd

data = pd.read_excel("Problem_list.xlsx", sheet_name="Sheet1")

# %%
# Clean up the revenue column (remove commas and convert to numeric)
df = data
df["revenue"] = df["revenue"].astype(str).str.replace(",", "").astype(float)

# Convert install_date to datetime
df["install_date"] = pd.to_datetime(df["install_date"])

df.sort_values("install_date", inplace=True)
df.reset_index(drop=True, inplace=True)
df.head()

# %%
geos = sorted(data["country"].unique())
channels = sorted(data["channel_id"].unique())
dates = pd.date_range(start=data["install_date"].min(), end=data["install_date"].max())
features = ["spend", "revenue", "clicks", "impressions", "installs"]
# Initialize GTCD tensor
GTCD = np.zeros((len(geos), len(dates), len(channels), len(features)))

# Mapping dictionaries
geo_idx = {geo: i for i, geo in enumerate(geos)}
date_idx = {date: i for i, date in enumerate(dates)}
channel_idx = {ch: i for i, ch in enumerate(channels)}

# Fill tensor
for _, row in data.iterrows():
    g = geo_idx[row["country"]]
    d = date_idx[row["install_date"]]
    c = channel_idx[row["channel_id"]]
    GTCD[g, d, c, :] += [row[feat] for feat in features]

GTCD.shape

# %%
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
import torch
import torch.nn as nn

# %%
### manual training
class NNNTransformer(nn.Module):
    def __init__(self, embedding_dim=5, nhead=1, num_layers=2, dim_feedforward=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        ## project input sequence format (G *T, C, D)
        self.input_projection = nn.Linear(embedding_dim, embedding_dim)

        encoder_layer = nn.TransformerEncoder(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )
        ## output layer: predict revenue (or any terget)
        self.output_head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        G, T, C, D = x.shape
        x = x.view(G * T, C, D)
        x = self.input_projection(x)

        ## apply Transformer
        x = self.transformer_encoder(x)  # (G*T, C, D)
        ## predict per channnel, or sum over channels
        x = self.output_head(x).squeeze(-1)  ## (G*T, C)
        x = x.mean(dim=1)  # average over channels --> shpe: (G*T,)
        return x.view(G, T)  # reshape to G, T

# %%
model = NNNTransformer()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# %%
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(key_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

# %%
import tensorflow as tf
from tensorflow.keras import layers, models


class NNNTransformer(tf.keras.Model):
    def __init__(self, embedding_dim=5, num_heads=1, num_layers=2, ff_dim=64):
        super(NNNTransformer, self).__init__()
        self.embedding_dim = embedding_dim

        self.input_projection = layers.Dense(embedding_dim)

        self.transformer_layers = [
            layers.TransformerBlock(
                num_heads=num_heads, key_dim=embedding_dim, ff_dim=ff_dim
            )
            for _ in range(num_layers)
        ]

        self.output_head = layers.Dense(1)

    def call(self, inputs):
        # inputs shape: (G, T, C, D)
        G, T, C, D = (
            tf.shape(inputs)[0],
            tf.shape(inputs)[1],
            tf.shape(inputs)[2],
            tf.shape(inputs)[3],
        )
        x = tf.reshape(inputs, [G * T, C, D])  # (G*T, C, D)
        x = self.input_projection(x)  # (G*T, C, D)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        x = self.output_head(x)  # (G*T, C, 1)
        x = tf.reduce_mean(x, axis=1)  # average over channels -> (G*T,)
        x = tf.reshape(x, [G, T])  # reshape back to (G, T)
        return x

# %%


# %%
# Dummy input (10 Geos, 51 Time steps, 1 Channel, 5 Dim embeddings)
x = tf.random.uniform((10, 51, 1, 5))
y = tf.random.uniform((10, 51))  # Dummy target revenue per geo & timestep

model = NNNTransformer()
model.compile(optimizer="adam", loss="mse")

# Wrap input as a tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

model.fit(dataset, epochs=10)

# %%


# %%


# %%
tensor = GTCD
# Prepare input and target
geo, time_steps, channels, features = tensor.shape
X = tensor.reshape((geo * channels, time_steps, features))
y = tensor[:, :, :, 1].sum(axis=1).flatten()  # Sum of revenue


# Build model
def build_nnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LayerNormalization()(inputs)
    x = MultiHeadAttention(num_heads=2, key_dim=features)(x, x)
    x = Dropout(0.1)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="linear")(x)
    return Model(inputs, outputs)


model = build_nnn_model((time_steps, features))
model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

# Train model
history = model.fit(X, y, epochs=50, batch_size=2, verbose=0)

# Return the last few loss values
history.history["loss"][-5:]

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
flat_GTCD = GTCD.reshape(-1, GTCD.shape[-1])  # (G*T*C, D)
GTCD_scaled = scaler.fit_transform(flat_GTCD).reshape(GTCD.shape)
GTCD_scaled

# %%
## convert tesnfor flow data:
import tensorflow as tf

# Convert to tf.Tensor
GTCD_tensor = tf.convert_to_tensor(GTCD_scaled, dtype=tf.float32)

# Dummy target: total revenue over time for each geo
target = tf.reduce_sum(
    GTCD_tensor[:, :, :, 1], axis=2
)  # use index 1 for revenue, shape = (G, T)

dataset = tf.data.Dataset.from_tensor_slices((GTCD_tensor, target)).batch(2)

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


class NNNTransformer(tf.keras.Model):
    def __init__(self, embedding_dim=5, num_heads=1, num_layers=2, ff_dim=64):
        super().__init__()
        self.proj = tf.keras.layers.Dense(embedding_dim)
        self.transformers = [
            TransformerBlock(num_heads, embedding_dim, ff_dim)
            for _ in range(num_layers)
        ]
        self.output_head = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # inputs: (G, T, C, D)
        G, T, C, D = (
            tf.shape(inputs)[0],
            tf.shape(inputs)[1],
            tf.shape(inputs)[2],
            tf.shape(inputs)[3],
        )
        x = tf.reshape(inputs, (G * T, C, D))  # (G*T, C, D)
        x = self.proj(x)  # (G*T, C, D)

        for layer in self.transformers:
            x = layer(x)

        x = self.output_head(x)  # (G*T, C, 1)
        x = tf.reduce_mean(x, axis=1)  # (G*T,)
        x = tf.reshape(x, (G, T))  # (G, T)
        return x

# %%
model = NNNTransformer()
model.compile(optimizer="adam", loss="mse")

model.fit(dataset, epochs=10)

# %%
# Normalize new GTCD input with the same scaler
flat_GTCD = GTCD.reshape(-1, GTCD.shape[-1])
GTCD_scaled = scaler.transform(flat_GTCD).reshape(GTCD.shape)

# Convert to tf.Tensor
GTCD_tensor = tf.convert_to_tensor(GTCD_scaled, dtype=tf.float32)

# %%
# Shape: (G, T, C, D)
predictions = model.predict(GTCD_tensor)

# %%
## predicted revenue per geo per time step
predictions

# %%
## Mapping Predictions to Pandas DataFrame:
import pandas as pd

# predictions: shape (len(geos), len(dates))
pred_df = pd.DataFrame(predictions, index=geos, columns=dates)
pred_df = pred_df.reset_index().melt(
    id_vars="index", var_name="install_date", value_name="predicted_revenue"
)
pred_df.rename(columns={"index": "country"}, inplace=True)

# %%
pred_df

# %%
## predict subset: (..., single geo)
single_geo_tensor = GTCD_tensor[
    geo_idx["FR"] : geo_idx["FR"] + 1
]  # shape: (1, T, C, D)
geo_pred = model.predict(single_geo_tensor)  # shape: (1, T)

# %%
geo_pred

# %%
# Make multip head prediction

# %%
# target: shape (G, T, D) — same as input, but as labels
target = GTCD_tensor  # shape: (G, T, C, D)
target = tf.reduce_mean(target, axis=2)  # average over channels → (G, T, D)

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
        G, T, C, D = (
            tf.shape(inputs)[0],
            tf.shape(inputs)[1],
            tf.shape(inputs)[2],
            tf.shape(inputs)[3],
        )
        x = tf.reshape(inputs, (G * T, C, D))  # (G*T, C, D)
        x = self.proj(x)

        for layer in self.transformers:
            x = layer(x)

        x = tf.reduce_mean(x, axis=1)  # (G*T, D)
        x = self.output_head(x)  # (G*T, 5)
        x = tf.reshape(x, (G, T, 5))  # (G, T, 5)
        return x

# %%
# Prepare dataset
target = tf.reduce_mean(GTCD_tensor, axis=2)  # shape: (G, T, 5)

dataset = tf.data.Dataset.from_tensor_slices((GTCD_tensor, target)).batch(2)

model = NNNTransformer(output_dim=5)
model.compile(optimizer="adam", loss="mse")
model.fit(dataset, epochs=10)

# %%
preds = model.predict(GTCD_tensor)  # shape: (G, T, 5)

# %%
# pred[:, :, 0] -> spend
# pred[:, :, 1] -> revenue
# pred[:, :, 2] -> clicks
# pred[:, :, 3] -> impressions
# pred[:, :, 4] -> installs
preds[:, :, 0]  ## --> spend

# %%
columns = ["spend", "revenue", "clicks", "impressions", "installs"]
results = []

for g_idx, geo in enumerate(geos):
    for t_idx, date in enumerate(dates):
        row = preds[g_idx, t_idx]
        results.append(
            {
                "country": geo,
                "install_date": date,
                **{col: row[i] for i, col in enumerate(columns)},
            }
        )

pred_df = pd.DataFrame(results)

# %%
pred_df

# %%
# which channel is driving performance (for each feature: spend, revenue, clicks, etc.).

# %%
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, return_attention_scores=True
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
        attn_output, attn_scores = self.att(
            x, x, return_attention_scores=True
        )  # (batch, heads, C, C)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        return self.norm2(x + ffn_output), attn_scores

# %%
import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.norm = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential(
            [layers.Dense(embed_dim * 4, activation="relu"), layers.Dense(embed_dim)]
        )
        self.dropout = layers.Dropout(0.1)

    def call(self, x, training=False, return_attention=False):
        attn_output, attn_scores = self.multihead_attn(
            x, x, return_attention_scores=True, training=training
        )
        x = self.norm(x + self.dropout(attn_output, training=training))
        ffn_output = self.ffn(x)
        x = self.norm(x + self.dropout(ffn_output, training=training))

        if return_attention:
            return x, attn_scores
        else:
            return x


class NNNTransformer(tf.keras.Model):
    def __init__(
        self,
        num_channels,
        num_features,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        output_dim=5,
    ):
        super().__init__()
        self.embed = layers.Dense(embed_dim)  # Project input features
        self.channel_pos_encoding = self.add_weight(
            name="channel_pos_encoding",
            shape=(1, num_channels, embed_dim),
            initializer="random_normal",
        )
        self.attn_blocks = [
            ChannelAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)
        ]
        self.global_pool = layers.GlobalAveragePooling1D()
        self.output_layer = layers.Dense(output_dim)  # Predict all 5 features

    def call(self, x, training=False, return_attention=False):
        # x: (G, T, C, F)
        B, T, C, F = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # Reshape to (G*T, C, F)
        x = tf.reshape(x, (-1, C, F))  # (B*T, C, F)
        x = self.embed(x)  # (B*T, C, E)
        x += self.channel_pos_encoding  # add positional encoding (learned)

        attn_scores_list = []
        for attn_block in self.attn_blocks:
            if return_attention:
                x, attn_scores = attn_block(x, training=training, return_attention=True)
                attn_scores_list.append(attn_scores)  # (B*T, H, C, C)
            else:
                x = attn_block(x, training=training)

        x = self.global_pool(x)  # (B*T, E)
        output = self.output_layer(x)  # (B*T, 5)
        output = tf.reshape(output, (B, T, -1))  # (B, T, 5)

        if return_attention:
            return output, attn_scores_list[-1]  # you can return all if needed
        else:
            return output

# %%
model = NNNTransformer(
    num_channels=GTCD_tensor.shape[2],
    num_features=GTCD_tensor.shape[3],
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    output_dim=5,
)

# %%
preds, attn_scores = model(GTCD_tensor, return_attention=True)
print(preds.shape)  # Should be (G, T, 5)
print(attn_scores.shape)  # Should be (G*T, H, C, C)

# %%
# Interpret attention per channel
# Average over heads
channel_attention = tf.reduce_mean(attn_scores, axis=1)  # (G*T, C, C)

# Let's say you want attention weights from channels to themselves:
channel_self_attn = tf.linalg.diag_part(channel_attention)  # (G*T, C)

# Or average across target dimension
channel_importance = tf.reduce_mean(channel_attention, axis=-1)  # (G*T, C)

# Optionally reshape back to (G, T, C)
G, T = GTCD_tensor.shape[0], GTCD_tensor.shape[1]
channel_importance = tf.reshape(channel_importance, (G, T, -1))

# %%
# preds, attn_scores = model(...)  # already done

# Calculate mean attention across batch
mean_attn = tf.reduce_mean(attn_scores, axis=0).numpy().flatten()

# Channel names (replace with real ones if you have them)
channel_names = [f"channel_{i}" for i in range(mean_attn.shape[0])]

# Create DataFrame
attn_df = pd.DataFrame(
    {"Channel": channel_names, "AttentionWeight": mean_attn}
).sort_values(by="AttentionWeight", ascending=False)

# Optional normalization
attn_df["Normalized"] = attn_df["AttentionWeight"] / attn_df["AttentionWeight"].sum()

print(attn_df)

# %%
Interpretation
These attention weights show how much each channel influenced the prediction on that date and geo. For example:

A high weight on channel 3 in France on Jan 20th → model thinks channel 3 is key that day.

You can extract top-N channels per day and plot trends over time to see shifting media impact.

Would you like help with:

Aggregating top contributing channels across time/geo?

Plotting attention heatmaps over time?

Using channel metadata (like video vs. static creative) to enhance interpretability?

Let me know how you want to go deeper 🙌

# %% [markdown]
# Interpretation
# These attention weights show how much each channel influenced the prediction on that date and geo. For example:
# 
# A high weight on channel 3 in France on Jan 20th → model thinks channel 3 is key that day.
# 
# You can extract top-N channels per day and plot trends over time to see shifting media impact.
# 

# %%
# Assuming these are defined:
geo_names = [f"geo_{i}" for i in range(GTCD_tensor.shape[0])]
platform_names = [f"platform_{i}" for i in range(GTCD_tensor.shape[0])]
channel_names = [f"channel_{i}" for i in range(GTCD_tensor.shape[2])]
output_features = ["spend", "revenue", "clicks", "impressions", "installs"]

# Run model
preds = model(GTCD_tensor)  # shape: (B, C, F)
preds_np = preds.numpy()

rows = []
for b in range(preds_np.shape[0]):
    for c in range(preds_np.shape[1]):
        row = {
            "Geo": geo_names[b],
            "Platform": platform_names[b],
            "Channel": channel_names[c],
        }
        for f in range(preds_np.shape[2]):
            row[output_features[f]] = preds_np[b, c, f]
        rows.append(row)

df = pd.DataFrame(rows)
print(df.head())

# %%
print("preds shape:", preds_np.shape)
print("len(channel_names):", len(channel_names))

# %%
print("preds shape:", preds_np.shape)
print("len(channel_names):", len(channel_names))

# %%
output_features = ["spend", "revenue", "clicks", "impressions", "installs"]
geo_names = ["US"]  # adjust if you have more
platform_names = ["iOS"]  # adjust if needed

channel_names = [f"channel_{i}" for i in range(preds_np.shape[1])]

rows = []
for b in range(preds_np.shape[0]):
    for c in range(preds_np.shape[1]):
        row = {
            "Geo": geo_names[b],
            "Platform": platform_names[b],
            "Channel": channel_names[c],
        }
        for f in range(preds_np.shape[2]):
            row[output_features[f]] = preds_np[b, c, f]
        rows.append(row)

df_preds = pd.DataFrame(rows)
print(df_preds.head())

# %%
## predict transition point;
# Sample shape: (1 geo/platform, T=10 days, 51 channels, 5 features)
# Just faking an array to show the logic
preds_np = np.random.rand(1, 10, 51, 5) * 100  # shape: (B, T, C, F)

output_features = ["spend", "revenue", "clicks", "impressions", "installs"]

# Feature indices
spend_idx = output_features.index("spend")
revenue_idx = output_features.index("revenue")

transition_points = []

for c in range(preds_np.shape[2]):  # over channels
    profit = preds_np[0, :, c, revenue_idx] - preds_np[0, :, c, spend_idx]
    transition_day = np.argmax(profit > 0) if np.any(profit > 0) else -1
    transition_points.append(
        {
            "Channel": f"channel_{c}",
            "TransitionDay": transition_day,
            "ROI": profit[transition_day] if transition_day != -1 else None,
        }
    )

transition_df = pd.DataFrame(transition_points)
print(transition_df.head())

# %%
# plot the revenue vs spend for some channels:

import matplotlib.pyplot as plt

channel_idx = 0
days = np.arange(preds_np.shape[1])

revenue = preds_np[0, :, channel_idx, revenue_idx]
spend = preds_np[0, :, channel_idx, spend_idx]

plt.plot(days, revenue, label="Revenue")
plt.plot(days, spend, label="Spend")
plt.title(f"Channel {channel_idx}")
plt.xlabel("Day")
plt.ylabel("Value")
plt.legend()
plt.show()

# %%
import numpy as np
import pandas as pd

# Feature indices
output_features = ["spend", "revenue", "clicks", "impressions", "installs"]
spend_idx = output_features.index("spend")
revenue_idx = output_features.index("revenue")

# Replace this with your actual predictions
# preds_np shape: (B, T, C, F)
# Example: (1 geo/platform, 10 days, 51 channels, 5 features)
B, T, C, F = preds_np.shape

results = []

for b in range(B):
    for c in range(C):
        spend = preds_np[b, :, c, spend_idx]
        revenue = preds_np[b, :, c, revenue_idx]

        cum_spend = np.cumsum(spend)
        cum_revenue = np.cumsum(revenue)

        transition_day = (
            np.argmax(cum_revenue > cum_spend)
            if np.any(cum_revenue > cum_spend)
            else -1
        )

        results.append(
            {
                "Geo": geo_names[b] if "geo_names" in locals() else f"geo_{b}",
                "Platform": (
                    platform_names[b]
                    if "platform_names" in locals()
                    else f"platform_{b}"
                ),
                "Channel": (
                    channel_names[c] if len(channel_names) == C else f"channel_{c}"
                ),
                "TransitionDay": transition_day,
                "FinalCumulativeRevenue": cum_revenue[-1],
                "FinalCumulativeSpend": cum_spend[-1],
                "ROI": (
                    (cum_revenue[-1] - cum_spend[-1]) / cum_spend[-1]
                    if cum_spend[-1] != 0
                    else np.nan
                ),
            }
        )

transition_df = pd.DataFrame(results)
transition_df = transition_df.sort_values(by="TransitionDay").reset_index(drop=True)
print(transition_df.head())

# %%
channel_idx = 0
days = np.arange(T)
spend = preds_np[0, :, channel_idx, spend_idx]
revenue = preds_np[0, :, channel_idx, revenue_idx]

plt.plot(days, np.cumsum(spend), label="Cumulative Spend")
plt.plot(days, np.cumsum(revenue), label="Cumulative Revenue")
plt.axvline(
    transition_df.loc[channel_idx, "TransitionDay"],
    color="red",
    linestyle="--",
    label="Transition",
)
plt.title(f"Channel {channel_idx}")
plt.xlabel("Day")
plt.ylabel("Cumulative Value")
plt.legend()
plt.show()

# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Prepare your last 3-day input ---
# Shape: (batch, time_steps=3, channels=1, features=5)
input_data = np.random.rand(1, 3, 1, 5).astype(np.float32)

# --- Step 2: Use your already-built model ---
# If the model is already defined as `model`, just use it
# Else uncomment the following (example dummy transformer model for context)
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(None, 1, 5)),
#     tf.keras.layers.Conv2D(32, kernel_size=(1, 1), activation='relu'),
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(5)
# ])

# --- Step 3: Predict step-by-step ---
n_future_days = 27
current_input = tf.convert_to_tensor(input_data)
predictions = []

for _ in range(n_future_days):
    pred = model(current_input, training=False)  # shape: (1, 3, 5)
    print("Prediction shape:", pred.shape)

    # Grab just the first predicted time step (shape: (1, 5))
    next_step = pred[:, 0, :]  # shape: (1, 5)

    # Reshape to match input format: (batch, time=1, channel=1, features)
    next_step = tf.reshape(next_step, (1, 1, 1, 5))

    # Append to input sequence
    current_input = tf.concat([current_input, next_step], axis=1)

    # Keep only last 3 time steps
    current_input = current_input[:, -3:, :, :]  # shape: (1, 3, 1, 5)

    # Collect predictions
    predictions.append(next_step.numpy())

# --- Step 4: Stack and plot ---
pred_stack = np.concatenate(predictions, axis=0).squeeze()  # shape: (27, 5)

# Plot the predicted features
plt.figure(figsize=(12, 6))
labels = ["spend", "revenue", "clicks", "impressions", "installs"]
for i in range(5):
    plt.plot(pred_stack[:, i], label=labels[i])
plt.title("Predicted Features Over Next 27 Days")
plt.xlabel("Day")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Convert predictions list to array of shape (n_future_days, 1, 1, 5)
predictions_array = np.concatenate(predictions, axis=1).squeeze()  # shape: (n_days, 5)

# Extract relevant features
revenue = predictions_array[:, 0]
spend = predictions_array[:, 1]
dayback = np.arange(len(revenue))

# Separate input and forecasted parts
input_days = np.arange(3)
pred_days = np.arange(3, len(revenue))

# Plotting
plt.figure(figsize=(10, 6))

# Revenue
plt.plot(input_days, revenue[:3], label="Revenue (Input)", color="blue", linestyle="-")
plt.plot(
    pred_days, revenue[3:], label="Revenue (Predicted)", color="blue", linestyle="--"
)

# Spend
plt.plot(input_days, spend[:3], label="Spend (Input)", color="green", linestyle="-")
plt.plot(pred_days, spend[3:], label="Spend (Predicted)", color="green", linestyle="--")

plt.xlabel("Dayback")
plt.ylabel("Amount")
plt.title("Revenue and Spend Over Forecasted Days")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%


# %%


# %%
import matplotlib.pyplot as plt
import numpy as np

# Convert predictions list to array of shape (n_days, 5)
predictions_array = np.concatenate(predictions, axis=1).squeeze()  # shape: (n_days, 5)

# Extract relevant features
revenue = predictions_array[:, 0]
spend = predictions_array[:, 1]
dayback = np.arange(len(revenue))

# Separate input (first 3) and predicted (rest)
input_days = dayback[:3]
pred_days = dayback[3:]

# Plotting
plt.figure(figsize=(10, 6))

# Revenue
plt.plot(input_days, revenue[:3], label="Spend (Input)", color="blue", linestyle="-")
plt.plot(
    pred_days, revenue[3:], label="Spend (Predicted)", color="blue", linestyle="--"
)
plt.plot(
    [input_days[-1], pred_days[0]],
    [revenue[2], revenue[3]],
    color="blue",
    linestyle="--",
)  # connect line

# Spend
plt.plot(input_days, spend[:3], label="Revenue (Input)", color="green", linestyle="-")
plt.plot(
    pred_days, spend[3:], label="Revenue (Predicted)", color="green", linestyle="--"
)
plt.plot(
    [input_days[-1], pred_days[0]], [spend[2], spend[3]], color="green", linestyle="--"
)  # connect line

# Vertical red line to show prediction start
plt.axvline(x=input_days[-1], color="red", linestyle=":", label="Prediction Start")

plt.xlabel("Next days")
plt.ylabel("Amount")
plt.title("Revenue and Spend Over Forecasted Days")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%



