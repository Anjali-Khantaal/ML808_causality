##################################################################
# STEP 4B: BAYESIAN RNN MODEL USING TENSORFLOW PROBABILITY
##################################################################
# NOTE: Execute this in a cell of your Jupyter/Colab notebook.
#       Make sure to have installed:
#         pip install --upgrade tensorflow tensorflow-probability
##################################################################

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfpl = tfp.layers

###############################################################################
# 1. Load and prepare the preprocessed data
###############################################################################
# We assume you've already run Steps 1-3 and have 'financial_timeseries_preprocessed.csv'.
# This file should contain:
#   - 'FedFunds_diff', 'SP500_log_return'
#   - plus lagged columns (if desired), but for an RNN we typically create
#     rolling windows rather than manually specifying lags.

data = pd.read_csv("financial_timeseries_preprocessed.csv", parse_dates=True, index_col=0)

# In a typical RNN scenario, we only need the two primary target columns
# and possibly some additional covariates. Here we keep it simple:
data = data[['FedFunds_diff', 'SP500_log_return']].copy()
data.dropna(inplace=True)  # Just in case

# Sort by date index if not already
data.sort_index(inplace=True)

print("Data shape:", data.shape)
print("Data columns:", data.columns.tolist())
print(data.head())


###############################################################################
# 2. Create windowed training data (X, Y) for the RNN
###############################################################################
#   We'll define a helper function that:
#     - Takes a sequence (data),
#     - A window length (seq_len),
#     - Returns arrays of shape:
#         X.shape -> (num_samples, seq_len, num_features)
#         Y.shape -> (num_samples, num_targets)
#     - We forecast only the next step (t+1) after seeing seq_len steps.
###############################################################################

def create_sequence_data(df, seq_len=12):
    """
    df: DataFrame or 2D array with shape (T, num_features).
    seq_len: Number of time steps to include in each input window.

    Returns:
       X: shape (num_samples, seq_len, num_features)
       Y: shape (num_samples, num_features)  # next-step predictions
    """
    values = df.values
    num_samples = len(values) - seq_len
    X_list, Y_list = [], []
    
    for i in range(num_samples):
        X_list.append(values[i : i+seq_len])
        Y_list.append(values[i+seq_len])  # next time step

    return np.array(X_list), np.array(Y_list)

# Define how many past months we look at:
SEQ_LEN = 12

# Build the dataset
X_all, Y_all = create_sequence_data(data, seq_len=SEQ_LEN)

print("X_all shape:", X_all.shape)  # (num_samples, SEQ_LEN, 2)
print("Y_all shape:", Y_all.shape)  # (num_samples, 2)


###############################################################################
# 3. Split data into train and test sets
###############################################################################
#   We'll do a simple chronological split (e.g., 80% train, 20% test).
###############################################################################
train_size = int(0.8 * len(X_all))

X_train = X_all[:train_size]
Y_train = Y_all[:train_size]

X_test = X_all[train_size:]
Y_test = Y_all[train_size:]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")


###############################################################################
# 4. Define a Bayesian RNN using TensorFlow Probability
###############################################################################
#   We'll use:
#     - tfp.layers.Convolution1DFlipout or tfp.layers.DenseFlipout
#       for Bayesian layers, but we want an RNN. TFP provides some
#       'Reparameterization' wrappers for LSTM/GRU. If not, we can do
#       a hybrid approach: regular LSTM for feature extraction, then
#       a DenseFlipout for the final distribution output.
#
#   For simplicity here, let's do:
#     - A standard LSTM layer (non-Bayesian) to handle sequences,
#     - Followed by a tfp.layers.DenseFlipout that outputs (loc, scale)
#       for a 2D distribution (FedFunds_diff, SP500_log_return).
###############################################################################

# We'll build a model that outputs a 2D Gaussian (mean + covariance).
# For simplicity, let's assume a diagonal covariance.

num_outputs = 2  # We want to predict 2 targets
event_size = num_outputs  # dimension of the output distribution

# We'll produce 2 (loc) + 2 (scale) = 4 linear outputs from the final layer
# Then interpret them as (loc=..., scale=...).
model = tf.keras.Sequential([
    # 1) LSTM for sequence feature extraction (non-Bayesian, keep it small).
    tf.keras.layers.LSTM(
        units=16,
        activation='tanh',
        return_sequences=False
    ),
    
    # 2) A Bayesian dense layer using Flipout, outputting parameters for the distribution
    tfpl.DenseFlipout(
        units=2 * event_size,  # 4 outputs: 2 for mean, 2 for log-scale
        activation=None
    ),
    
    # 3) Convert these parameters into a DistributionLambda that returns
    #    a MultivariateNormalDiag distribution over the 2D output space.
    tfpl.DistributionLambda(
        make_distribution_fn=lambda params: tfd.MultivariateNormalDiag(
            loc=params[..., :event_size],
            scale_diag=1e-3 + tf.nn.softplus(params[..., event_size:])
        )
    )
])

# Check model summary
model.build(input_shape=(None, SEQ_LEN, 2))
model.summary()


###############################################################################
# 5. Define the negative log likelihood loss and compile the model
###############################################################################
#   We'll use the standard nll: -log p(y|x).
###############################################################################
def negloglik(y_true, y_pred):
    """y_pred is a tfd.Distribution, we return -log_prob of y_true."""
    return -y_pred.log_prob(y_true)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=negloglik
)

###############################################################################
# 6. Fit the model and monitor for overfitting
###############################################################################
#   We can use early stopping to reduce overfitting risk.
###############################################################################
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,  # if no improvement for 10 epochs, stop
    restore_best_weights=True
)

history = model.fit(
    x=X_train,
    y=Y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Plot training and validation loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Negative Log Likelihood")
plt.legend()
plt.show()


###############################################################################
# 7. Evaluate on the test set
###############################################################################
test_loss = model.evaluate(X_test, Y_test)
print(f"Test set Negative Log Likelihood: {test_loss:.4f}")

# Predict distributions
yhat_dist = model(X_test)  # This is a tfd.MultivariateNormalDiag
yhat_means = yhat_dist.mean().numpy()
yhat_scales = yhat_dist.stddev().numpy()

print("\nSample predictions:")
for i in range(3):
    print(f" True Y: {Y_test[i]}, Pred Mean: {yhat_means[i]}, Pred Stddev: {yhat_scales[i]}")

# Done! We now have a Bayesian RNN model that returns a predictive distribution
# for the next time step's FedFunds_diff and SP500_log_return.

###############################################################################
# 8. Notes on avoiding overfit/underfit
###############################################################################
# - We used a modest LSTM size (units=16) and early stopping.
# - Flipout layers help incorporate weight uncertainty, typically
#   preventing extreme overfitting (but you can tune prior scales).
# - If underfitting, consider more LSTM layers/units or smaller LR.
# - If overfitting, reduce units, add dropout, or strengthen priors.
