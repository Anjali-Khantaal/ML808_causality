"""
Step 6: Enhanced Causal Generative Model Development
------------------------------------------------------
This script extends the baseline generative model by incorporating causal conditioning.
We design a “Causal VAE” with an LSTM encoder and a modular decoder that follows the causal order:
    1. Generate FEDFUNDS_Level (exogenous) sequence.
    2. Generate Macro sequence [GS10_Change, Inflation_YoY] conditioned on FED.
    3. Generate SP500_Returns sequence conditioned on Macro.
The final output is reassembled into original order:
    [SP500_Returns, GS10_Change, FEDFUNDS_Level, Inflation_YoY].

Improvements in this version:
  - All 4 variables are normalized (using the training set statistics).
  - A validation loop is incorporated and training/validation losses are plotted and saved.
  - Teacher forcing with scheduled sampling (decaying over epochs) is applied.
  - KL annealing is implemented.
  - Early stopping is added to stop training if the validation loss does not improve for a given patience.
  - Detailed loss monitoring is in place.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

# Set seeds for reproducibility.
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---------------------------
# 1. Data Preparation and Normalization
# ---------------------------
data_df = pd.read_csv("final_processed_data.csv", index_col=0, parse_dates=True)
print("Original processed data shape:", data_df.shape)

# Use the 4 key variables: [SP500_Returns, GS10_Change, FEDFUNDS_Level, Inflation_YoY]
features = ['SP500_Returns', 'GS10_Change', 'FEDFUNDS_Level', 'Inflation_YoY']
data_df = data_df[features]
data_values = data_df.values  # shape: (num_samples, 4)

# Create overlapping sequences (sliding window)
seq_length = 12
num_samples = data_values.shape[0] - seq_length + 1
print("Creating sequences of length:", seq_length)
sequences = [data_values[i : i + seq_length] for i in range(num_samples)]
sequences = np.array(sequences)
print("Total sequences created:", sequences.shape[0])

# Split into train (80%) and validation (20%)
split_idx = int(0.8 * sequences.shape[0])
train_sequences = sequences[:split_idx]
val_sequences = sequences[split_idx:]
print("Training sequences shape:", train_sequences.shape)
print("Validation sequences shape:", val_sequences.shape)

# Normalize using statistics from the training set
# Compute per-feature mean and std (reshape to allow broadcasting)
train_flat = train_sequences.reshape(-1, len(features))
feature_means = train_flat.mean(axis=0)
feature_stds  = train_flat.std(axis=0)
print("Feature means:", feature_means)
print("Feature stds:", feature_stds)

def normalize(seq):
    # seq shape: (num_seq, seq_length, num_features)
    return (seq - feature_means) / feature_stds

train_sequences = normalize(train_sequences)
val_sequences = normalize(val_sequences)

# Define PyTorch Dataset.
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
    def __len__(self):
        return self.sequences.shape[0]
    def __getitem__(self, idx):
        return self.sequences[idx], self.sequences[idx]

batch_size = 32
train_dataset = TimeSeriesDataset(train_sequences)
val_dataset = TimeSeriesDataset(val_sequences)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------
# 2. Define the Enhanced Causal LSTM-VAE Model with Teacher Forcing
# ---------------------------
class CausalLSTM_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_length, num_layers=1):
        super(CausalLSTM_VAE, self).__init__()
        self.input_dim = input_dim      # 4
        self.hidden_dim = hidden_dim    # e.g., 128
        self.latent_dim = latent_dim    # e.g., 32
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # Encoder: Standard LSTM encoder.
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder modules:
        # 1. FED Decoder (for FEDFUNDS_Level, dim=1), input: (1 + latent_dim)
        self.decoder_fed_lstm = nn.LSTM(input_size=1 + latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder_fed_out = nn.Linear(hidden_dim, 1)
        
        # 2. Macro Decoder (for [GS10_Change, Inflation_YoY], dim=2), input: (1 + latent_dim)
        self.decoder_macro_lstm = nn.LSTM(input_size=1 + latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder_macro_out = nn.Linear(hidden_dim, 2)
        
        # 3. Stock Decoder (for SP500_Returns, dim=1), input: (1 + latent_dim + 2)
        self.decoder_stock_lstm = nn.LSTM(input_size=1 + latent_dim + 2, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder_stock_out = nn.Linear(hidden_dim, 1)
    
    def encode(self, x):
        outputs, (h_n, c_n) = self.encoder_lstm(x)
        h_final = h_n[-1]  # (batch, hidden_dim)
        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, fed_target=None, macro_target=None, stock_target=None, teacher_forcing_ratio=0.0):
        batch_size = z.size(0)
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_length, 1)
        
        # Stage 1: FED Decoder
        fed_in = torch.zeros(batch_size, 1, 1, device=z.device)
        fed_outputs = []
        hidden_fed = None
        for t in range(self.seq_length):
            fed_input_t = torch.cat([fed_in, z_repeated[:, t:t+1, :]], dim=-1)
            out_fed, hidden_fed = self.decoder_fed_lstm(fed_input_t, hidden_fed)
            fed_out_t = self.decoder_fed_out(out_fed)
            fed_outputs.append(fed_out_t)
            if fed_target is not None and t < self.seq_length and random.random() < teacher_forcing_ratio:
                fed_in = fed_target[:, t:t+1, :]
            else:
                fed_in = fed_out_t
        fed_hat = torch.cat(fed_outputs, dim=1)
        
        # Stage 2: Macro Decoder
        macro_in = torch.zeros(batch_size, 1, 1, device=z.device)
        macro_outputs = []
        hidden_macro = None
        for t in range(self.seq_length):
            macro_input_t = torch.cat([macro_in, z_repeated[:, t:t+1, :]], dim=-1)
            out_macro, hidden_macro = self.decoder_macro_lstm(macro_input_t, hidden_macro)
            macro_out_t = self.decoder_macro_out(out_macro)
            macro_outputs.append(macro_out_t)
            if macro_target is not None and t < self.seq_length and random.random() < teacher_forcing_ratio:
                macro_in = macro_target[:, t:t+1, 0:1]
            else:
                macro_in = macro_out_t[:, :, 0:1]
        macro_hat = torch.cat(macro_outputs, dim=1)
        
        # Stage 3: Stock Decoder
        stock_in = torch.zeros(batch_size, 1, 1, device=z.device)
        stock_outputs = []
        hidden_stock = None
        for t in range(self.seq_length):
            macro_t = macro_hat[:, t:t+1, :]
            stock_input_t = torch.cat([stock_in, z_repeated[:, t:t+1, :], macro_t], dim=-1)
            out_stock, hidden_stock = self.decoder_stock_lstm(stock_input_t, hidden_stock)
            stock_out_t = self.decoder_stock_out(out_stock)
            stock_outputs.append(stock_out_t)
            if stock_target is not None and t < self.seq_length and random.random() < teacher_forcing_ratio:
                stock_in = stock_target[:, t:t+1, :]
            else:
                stock_in = stock_out_t
        stock_hat = torch.cat(stock_outputs, dim=1)
        
        # Final assembly (original order: [SP500_Returns, GS10_Change, FEDFUNDS_Level, Inflation_YoY])
        output = torch.cat([stock_hat, macro_hat[:, :, 0:1], fed_hat, macro_hat[:, :, 1:2]], dim=-1)
        return output, fed_hat, macro_hat, stock_hat, z
    
    def forward(self, x, teacher_forcing_ratio=0.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        fed_target = x[:, :, 2:3]
        macro_target = x[:, :, [1, 3]]
        stock_target = x[:, :, 0:1]
        x_recon, fed_hat, macro_hat, stock_hat, z = self.decode(z, fed_target, macro_target, stock_target, teacher_forcing_ratio)
        return x_recon, mu, logvar, fed_hat, macro_hat, stock_hat

# ---------------------------
# 3. Define Loss Function
# ---------------------------
def loss_function_causal(recon_x, x, mu, logvar, fed_hat, macro_hat, stock_hat, kl_weight=1.0):
    gt_fed = x[:, :, 2:3]
    gt_macro = x[:, :, [1, 3]]
    gt_stock = x[:, :, 0:1]
    
    loss_fed = F.mse_loss(fed_hat, gt_fed, reduction='sum')
    loss_macro = F.mse_loss(macro_hat, gt_macro, reduction='sum')
    loss_stock = F.mse_loss(stock_hat, gt_stock, reduction='sum')
    recon_loss = loss_fed + loss_macro + loss_stock
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss, loss_fed, loss_macro, loss_stock

# ---------------------------
# 4. Training Setup and Hyperparameters
# ---------------------------
input_dim = 4
hidden_dim = 128       # Increased capacity.
latent_dim = 32        # Increased capacity.
num_layers = 1
num_epochs = 100        # Maximum epochs; early stopping may stop training earlier.
learning_rate = 0.001
patience = 10          # Early stopping patience.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CausalLSTM_VAE(input_dim, hidden_dim, latent_dim, seq_length, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# KL Annealing: linearly increase KL weight until mid-training.
def get_kl_weight(epoch, num_epochs, max_weight=1.0, anneal_fraction=0.7):
    if epoch < num_epochs * anneal_fraction:
        return max_weight * (epoch / (num_epochs * anneal_fraction))
    else:
        return max_weight

# Teacher Forcing: initial ratio decays linearly.
def get_teacher_forcing_ratio(epoch, num_epochs, initial_ratio=0.8, decay_epochs=70):
    return initial_ratio * max(0, (decay_epochs - epoch) / decay_epochs)

# Lists to store loss values for plotting.
train_losses = []
val_losses = []

best_val_loss = float('inf')
epochs_no_improve = 0

# ---------------------------
# 5. Training Loop with Validation and Early Stopping
# ---------------------------
print("Starting training of Enhanced Causal VAE...")
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    train_recon = 0.0
    train_kl = 0.0
    for batch_idx, (batch_data, _) in enumerate(train_loader):
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        tf_ratio = get_teacher_forcing_ratio(epoch, num_epochs, initial_ratio=0.8)
        kl_weight = get_kl_weight(epoch, num_epochs, max_weight=1.0, anneal_fraction=0.5)
        recon_x, mu, logvar, fed_hat, macro_hat, stock_hat = model(batch_data, teacher_forcing_ratio=tf_ratio)
        loss, recon_loss, kl_loss, _, _, _ = loss_function_causal(recon_x, batch_data, mu, logvar, fed_hat, macro_hat, stock_hat, kl_weight)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_recon = train_recon / len(train_loader.dataset)
    avg_train_kl = train_kl / len(train_loader.dataset)
    
    # Validation loop.
    model.eval()
    val_loss = 0.0
    val_recon = 0.0
    val_kl = 0.0
    with torch.no_grad():
        for batch_idx, (batch_data, _) in enumerate(val_loader):
            batch_data = batch_data.to(device)
            recon_x, mu, logvar, fed_hat, macro_hat, stock_hat = model(batch_data, teacher_forcing_ratio=0.0)
            loss, recon_loss, kl_loss, _, _, _ = loss_function_causal(recon_x, batch_data, mu, logvar, fed_hat, macro_hat, stock_hat, kl_weight)
            val_loss += loss.item()
            val_recon += recon_loss.item()
            val_kl += kl_loss.item()
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_recon = val_recon / len(val_loader.dataset)
    avg_val_kl = val_kl / len(val_loader.dataset)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch}/{num_epochs} — TF Ratio: {tf_ratio:.4f}, KL Weight: {kl_weight:.4f}")
    print(f"   Training Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f})")
    print(f"   Validation Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f})")
    
    # Early stopping: if validation loss did not improve.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Save the best model.
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Validation loss did not improve for {patience} epochs. Early stopping.")
            break

# Plot and save the training and validation losses.
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
plt.show()

# If early stopping was triggered, load the best model.
if epochs_no_improve >= patience:
    model.load_state_dict(best_model_state)
    print("Best model loaded based on validation loss.")

print("Training complete.")

# ---------------------------
# 6. Generate Synthetic Data (Enhanced Model)
# ---------------------------
model.eval()
with torch.no_grad():
    num_generate = 5
    z_sample = torch.randn(num_generate, latent_dim).to(device)
    synthetic_full, _, _, _, _ = model.decode(z_sample, teacher_forcing_ratio=0.0)
    synthetic_full = synthetic_full.cpu().numpy()
print("Enhanced synthetic sequences generated. Shape:", synthetic_full.shape)

# ---------------------------
# 7. Debugging & Evaluation
# ---------------------------
real_seq = train_sequences[0]  # using a training sample (normalized)
synthetic_seq = synthetic_full[0]

print("\nReal Sequence Statistics (per feature):")
print("Mean:", np.mean(real_seq, axis=0))
print("Std:", np.std(real_seq, axis=0))

print("\nSynthetic Sequence Statistics (per feature):")
print("Mean:", np.mean(synthetic_seq, axis=0))
print("Std:", np.std(synthetic_seq, axis=0))

time_axis = np.arange(seq_length)
plt.figure(figsize=(12, 8))
for i, fname in enumerate(features):
    plt.subplot(2, 2, i+1)
    plt.plot(time_axis, real_seq[:, i], label="Real")
    plt.plot(time_axis, synthetic_seq[:, i], label="Synthetic", linestyle="--")
    plt.title(fname)
    plt.xlabel("Time step")
    plt.ylabel("Normalized Value")
    plt.legend()
plt.tight_layout()
plt.savefig("real_vs_synthetic_comparison.png")
plt.show()

# ---------------------------
# 8. Save Enhanced Model & Synthetic Data
# ---------------------------
enhanced_model_path = "./causal_lstm_vae_model.pth"
torch.save(model.state_dict(), enhanced_model_path)
print("Enhanced model saved to:", enhanced_model_path)

enhanced_synthetic_path = "./enhanced_synthetic_sequences.npy"
np.save(enhanced_synthetic_path, synthetic_full)
print("Enhanced synthetic sequences saved to:", enhanced_synthetic_path)

print("Step 6: Enhanced Causal Generative Model Development Complete.")
