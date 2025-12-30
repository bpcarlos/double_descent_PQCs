import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load saved data
with open("experiment_results.pkl", "rb") as f:
    data = pickle.load(f)

# Unpack what we need
params = np.array(data["params"])
mean_losses = np.array(data["mean_losses"])
std_losses = np.array(data["std_losses"])
mean_eig = np.array(data["mean_eig"])
std_eig = np.array(data["std_eig"])
mean_eig_init = np.array(data["mean_eig_init"])
std_eig_init = np.array(data["std_eig_init"])
peak_pos = data["peak_pos"]
total_param_arr = np.array(data["total_param_arr"])
K = data["K"]
n_train = data["n_train"]

# === Recreate your main plot ===
fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:red'
ax1.set_xlabel("Number of Parameters")
ax1.set_ylabel("Test Loss", color=color)
ax1.plot(params, mean_losses, marker='o', color=color, label="Population loss")
ax1.fill_between(params, mean_losses - std_losses, mean_losses + std_losses, alpha=0.2, color=color)
ax1.tick_params(axis='y')
ax1.legend(fontsize=14)

color = 'tab:blue'
ax2 = ax1.twinx()
ax2.plot(params, mean_eig, marker='o', color=color, label='Min. non-zero ev')
ax2.fill_between(params, mean_eig - std_eig, mean_eig + std_eig, alpha=0.2, color=color)
ax2.set_yscale('log')
ax2.plot(total_param_arr, mean_eig_init, marker='o', color='lightblue', label='Min. non-zero ev (init)')
ax2.fill_between(params, mean_eig_init - std_eig_init, mean_eig_init + std_eig_init, alpha=0.2, color='lightblue')
ax2.tick_params(axis='y')
ax2.legend(fontsize=14)

ax1.axvline(x=peak_pos, color='gray', linestyle='--', label=f'Interpolation Threshold (K·N = {K*n_train})')
plt.grid(True)
plt.tight_layout()
plt.legend(fontsize=14)
plt.show()
