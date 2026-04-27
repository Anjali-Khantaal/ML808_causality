"""
Step 5: Baseline Generative Model Development
------------------------------------------------
This script implements a baseline generative model using a Variational Autoencoder (VAE)
with an LSTM encoder-decoder. The purpose is to generate synthetic time-series data for our
four variables: SP500_Returns, GS10_Change, FEDFUNDS_Level, and Inflation_YoY.

Outputs/debug prints are provided to facilitate sanity checks, and sample plots compare
synthetic with real data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Set random seeds for reproducibility.
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------
# 1. Data Preparation
# ---------------------------
# Load the preprocessed data saved as "final_processed_data.csv" (from Step 3)
data_df = pd.read_csv("final_processed_data.csv", index_col=0, parse_dates=True)
print("Original processed data shape:", data_df.shape)

# We use the 4 variables for the generative model
features = ['SP500_Returns', 'GS10_Change', 'FEDFUNDS_Level', 'Inflation_YoY']
data_df = data_df[features]
data_values = data_df.values  # shape: (num_samples, 4)

# Define a fixed sequence length for our time-series (e.g., 12 months)
seq_length = 12  
num_samples = data_values.shape[0] - seq_length + 1
print("Creating sequences of length:", seq_length)

# Create overlapping sequences (using a sliding window)
sequences = []
for i in range(num_samples):
    seq = data_values[i : i + seq_length]
    sequences.append(seq)
sequences = np.array(sequences)  # shape: (num_sequences, seq_length, 4)
print("Total sequences created:", sequences.shape[0])

# Split the sequences into train and validation sets (e.g., 80% train, 20% val)
split_idx = int(0.8 * sequences.shape[0])
train_sequences = sequences[:split_idx]
val_sequences = sequences[split_idx:]
print("Training sequences shape:", train_sequences.shape)
print("Validation sequences shape:", val_sequences.shape)

# Define a PyTorch Dataset for our sequences.
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        # Convert to float32 and then to torch tensors.
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        
    def __len__(self):
        return self.sequences.shape[0]
    
    def __getitem__(self, idx):
        # For an autoencoder, input and target are the same sequence.
        return self.sequences[idx], self.sequences[idx]

batch_size = 32
train_dataset = TimeSeriesDataset(train_sequences)
val_dataset = TimeSeriesDataset(val_sequences)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------
# 2. Define the LSTM-VAE Model
# ---------------------------
class LSTM_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(LSTM_VAE, self).__init__()
        self.input_dim = input_dim      # Here, 4
        self.hidden_dim = hidden_dim    # e.g., 64
        self.latent_dim = latent_dim    # e.g., 16
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Map hidden state to latent space: compute mean and logvar
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: project latent to initial hidden state for LSTM decoder
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Output layer: map hidden state to output (same as input_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        # x: (batch, seq_length, input_dim)
        outputs, (h_n, c_n) = self.encoder_lstm(x)
        # Use the final hidden state h_n from the last layer. Shape: (num_layers, batch, hidden_dim)
        h_final = h_n[-1]  # shape: (batch, hidden_dim)
        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_len):
        # Initialize hidden state from latent vector z.
        h0 = self.fc_hidden(z).unsqueeze(0).repeat(self.num_layers, 1, 1)  # shape: (num_layers, batch, hidden_dim)
        c0 = torch.zeros_like(h0)
        # Prepare a start-of-sequence token (can be zeros)
        batch_size = z.size(0)
        # We'll feed zeros as initial input (or you could use teacher forcing during training).
        decoder_input = torch.zeros(batch_size, 1, self.input_dim, device=z.device)
        outputs = []
        # Generate sequence step-by-step
        hidden = (h0, c0)
        for _ in range(seq_len):
            out, hidden = self.decoder_lstm(decoder_input, hidden)
            out_step = self.output_layer(out)  # shape: (batch, 1, input_dim)
            outputs.append(out_step)
            # For simplicity, use the output as the next input (autoregressive generation)
            decoder_input = out_step
        outputs = torch.cat(outputs, dim=1)  # shape: (batch, seq_len, input_dim)
        return outputs
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, x.size(1))
        return x_recon, mu, logvar

# Hyperparameters
input_dim = 4
hidden_dim = 64
latent_dim = 16
num_layers = 1
num_epochs = 50
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = LSTM_VAE(input_dim, hidden_dim, latent_dim, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---------------------------
# 3. Define the Loss Function
# ---------------------------
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss: Mean Squared Error over sequences
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

# ---------------------------
# 4. Training Loop
# ---------------------------
print("Starting training...")
model.train()
for epoch in range(1, num_epochs + 1):
    train_loss = 0.0
    recon_loss_epoch = 0.0
    kl_loss_epoch = 0.0
    for batch_idx, (batch_data, _) in enumerate(train_loader):
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch_data)
        loss, recon_loss, kl_loss = loss_function(recon_batch, batch_data, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        recon_loss_epoch += recon_loss.item()
        kl_loss_epoch += kl_loss.item()
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_loss_epoch / len(train_loader.dataset)
    avg_kl = kl_loss_epoch / len(train_loader.dataset)
    print(f"Epoch {epoch}/{num_epochs} — Avg Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")

print("Training complete.")

# ---------------------------
# 5. Generate Synthetic Data
# ---------------------------
model.eval()
# For generation, we sample latent vectors from a standard normal distribution.
with torch.no_grad():
    num_generate = 5  # Generate 5 sample sequences for demonstration.
    z_sample = torch.randn(num_generate, latent_dim).to(device)
    synthetic_sequences = model.decode(z_sample, seq_length)  # shape: (num_generate, seq_length, input_dim)
    synthetic_sequences = synthetic_sequences.cpu().numpy()

print("Synthetic sequences generated. Shape:", synthetic_sequences.shape)

# ---------------------------
# 6. Evaluation & Debugging Outputs
# ---------------------------
# Compute summary statistics on one synthetic sequence versus one real sequence.
real_seq = train_sequences[0]  # select first training sequence
synthetic_seq = synthetic_sequences[0]

print("\nReal Sequence Statistics:")
print("Mean per feature:", np.mean(real_seq, axis=0))
print("Std per feature:", np.std(real_seq, axis=0))

print("\nSynthetic Sequence Statistics:")
print("Mean per feature:", np.mean(synthetic_seq, axis=0))
print("Std per feature:", np.std(synthetic_seq, axis=0))

# Plot sample real vs. synthetic sequences (one feature at a time)
time_axis = np.arange(seq_length)
feature_names = features  # same order as input

plt.figure(figsize=(12, 8))
for i, fname in enumerate(feature_names):
    plt.subplot(2, 2, i+1)
    plt.plot(time_axis, real_seq[:, i], label="Real")
    plt.plot(time_axis, synthetic_seq[:, i], label="Synthetic", linestyle="--")
    plt.title(fname)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------
# 7. Save the Trained Model and Generated Samples
# ---------------------------
model_save_path = "./lstm_vae_model.pth"
torch.save(model.state_dict(), model_save_path)
print("Trained model saved to:", model_save_path)

synthetic_save_path = "./synthetic_sequences.npy"
np.save(synthetic_save_path, synthetic_sequences)
print("Synthetic sequences saved to:", synthetic_save_path)

print("Step 5: Baseline Generative Model Development Complete.")
