import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, explained_variance_score

# =========================================================
# 1) COMPLEX PREDICTION: Multi-Step "Lookback" Forecasting
# =========================================================
# Instead of H[t] -> X[t+1], we use [H[t], H[t-1], H[t-2]] -> X[t+1]
# This captures momentum and seasonal trends.

def create_lookback_dataset(data_h, target_x, lookback=3):
    X_train, Y_train = [], []
    for i in range(lookback, len(data_h)):
        # Flatten the last 'n' days of atoms into a single feature vector
        window = data_h[i-lookback:i].flatten()
        X_train.append(window)
        Y_train.append(target_x[i])
    return np.array(X_train), np.array(Y_train)

lookback = 5 # Using a 5-day window to predict the 6th day
H_complex, y_complex = create_lookback_dataset(sparse_representation, X_scaled, lookback)

# Minimize Error using a more aggressive Ridge penalty for the high-dimensional input
complex_forecaster = Ridge(alpha=5.0).fit(H_complex, y_complex)

# Recursive Forecast with Lookback
forecast_steps = 21
history = list(sparse_representation[-lookback:]) # Start with last known window
forecast_results = []

for _ in range(forecast_steps):
    current_window = np.array(history[-lookback:]).flatten().reshape(1, -1)
    pred_scaled = complex_forecaster.predict(current_window)
    forecast_results.append(pred_scaled[0])
    
    # Update H: Transform prediction back to atom space and slide the window
    new_h = dict_learner.transform(pred_scaled)
    history.append(new_h[0])

df_forecast = pd.DataFrame(scaler.inverse_transform(np.array(forecast_results)), columns=features)

# =========================================================
# 2) COMPLEX GENERATION: Stochastic Random Walk (Monte Carlo)
# =========================================================
# Instead of a linear morph, we simulate a "weather system" that 
# evolves with random shocks, staying within the bounds of learned atoms.

def generate_stochastic_season(n_days=90, volatility=0.2):
    # Start with a neutral state (average of all atoms)
    current_h = np.mean(sparse_representation, axis=0)
    synthetic_h = []
    
    for _ in range(n_days):
        # Add a "Random Walk" shock
        shock = np.random.normal(0, volatility, size=n_atoms)
        current_h = np.clip(current_h + shock, 0, None) # Atoms stay non-negative
        
        # Soft-normalization: Prevent atoms from exploding
        current_h = current_h / (np.sum(current_h) + 1e-6) 
        synthetic_h.append(current_h)
    
    # Project back to physical space
    gen_scaled = np.dot(np.array(synthetic_h), dictionary_atoms)
    return pd.DataFrame(scaler.inverse_transform(gen_scaled), columns=features)

df_stochastic = generate_stochastic_season(n_days=90, volatility=0.15)

# =========================================================
# 3) VISUALIZATION & INTERPRETATION
# =========================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Forecast Plotting
ax1.plot(df_forecast['temp_max'], 'r-^', label='Predicted Max Temp (Lookback=5)')
ax1.plot(df_forecast['wind'], 'g--', label='Predicted Wind')
ax1.set_title("Complex Multi-Day Forecast (Captures Temporal Momentum)")
ax1.legend()
ax1.grid(True, alpha=0.2)

# Stochastic Generation Plotting
ax2.plot(df_stochastic['temp_max'], color='orange', alpha=0.7, label='Synthetic Temp')
ax2.fill_between(range(90), df_stochastic['temp_min'], df_stochastic['temp_max'], color='orange', alpha=0.1)
ax2.bar(range(90), df_stochastic['precipitation'], color='blue', alpha=0.3, label='Synthetic Rain')
ax2.set_title("Stochastic 90-Day Simulation (Random Walk in Latent Space)")
ax2.legend()

plt.tight_layout()
plt.show()

# Metrics
print(f"Explained Variance (Complex Model): {explained_variance_score(y_complex, complex_forecaster.predict(H_complex)):.4f}")