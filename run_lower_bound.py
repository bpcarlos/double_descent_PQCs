"""
run_lower_bound.py
------------------
Run this file AFTER your main experiment has finished and saved
experiment_results.pkl.

It loads your saved results, reloads one trained model per depth,
computes the Theorem 8 lower bound, and plots everything.

Usage:
    python run_lower_bound.py

Requirements:
    - experiment_results.pkl must exist in the same folder
    - your main script must be importable as a module (see note below)

NOTE: rename your main script to e.g. quantum_experiment.py and make
sure it does NOT run automatically on import. The easiest way is to
wrap the experiment loop at the bottom in:

    if __name__ == "__main__":
        # ... all your experiment code ...

Then this file can do:  from quantum_experiment import *
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Import everything from your existing script ──────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Regression", "21"))
from main_code import (
    create_quantum_layer,
    QuantumModel,
    compute_ZS,
    gauss_newton_mse_from_tensor,
    min_nonzero_eigval,
    load_synthetic_multiclass_linear,
    n_qubits,
    depths,
    selected_classes,
    c,
    K,
    n_train,
)

# ── Theorem 8 lower bound function ───────────────────────────────────

def compute_theorem8_lower_bound(
    model,
    X_train, y_train,
    X_test,  y_test,
    alpha=1.0,
    tol_eigval=1e-10,
    tol_sigma=1e-3,
):
    """
    Theorem 8 lower bound (A32):
      L(θ̂_S) >= L'(S) + 1/(N+1) * α * σ²_min * λ_min(C_f(θ̂_S)) / λ_min(Ĥ_S(θ̂_S))
    """
    model.eval()
    N = X_train.shape[0]

    # L'(S) — training loss
    with torch.no_grad():
        train_loss = nn.MSELoss()(model(X_train), y_train).item()

    # σ²_min — for MSE: ∇_f ℓ = f(x)-y, so σ²_z = ‖f(x)-y‖² per test sample
    with torch.no_grad():
        residuals = model(X_test) - y_test
        sigma2 = (residuals ** 2).sum(dim=1).cpu().numpy()
    mask = sigma2 >= tol_sigma * sigma2.mean()
    sigma2_min = float(sigma2[mask].min() if mask.any() else sigma2.min())

    # λ_min(C_f) — min non-zero eigval of (1/M) Z_test^T Z_test
    Z_test = compute_ZS(model, X_test).detach().cpu().numpy()
    C_f = (Z_test.T @ Z_test) / X_test.shape[0]
    ev_Cf = np.linalg.eigvalsh(C_f)
    nz_Cf = ev_Cf[ev_Cf > tol_eigval]
    lambda_min_Cf = float(nz_Cf.min()) if nz_Cf.size > 0 else 0.0

    # λ_min(Ĥ_S) — min non-zero eigval of outer-product Hessian on training set
    H_o = gauss_newton_mse_from_tensor(model, X_train)
    lambda_r_H, _ = min_nonzero_eigval(H_o, tol=tol_eigval)

    complexity = (0.0 if lambda_r_H == 0.0
                  else (1.0 / (N + 1)) * alpha * sigma2_min * lambda_min_Cf / lambda_r_H)

    return {
        "lower_bound":     train_loss + complexity,
        "train_loss":      train_loss,
        "sigma2_min":      sigma2_min,
        "lambda_min_Cf":   lambda_min_Cf,
        "lambda_r_H":      lambda_r_H,
        "complexity_term": complexity,
    }


# ── Load saved experiment results ────────────────────────────────────

if not os.path.exists("experiment_results.pkl"):
    raise FileNotFoundError(
        "experiment_results.pkl not found.\n"
        "Run  python Regression/21/main_code.py  first to generate it, "
        "then re-run this script."
    )

with open("experiment_results.pkl", "rb") as f:
    data = pickle.load(f)

params        = data["params"]
mean_losses   = data["mean_losses"]
std_losses    = data["std_losses"]
mean_eig      = data["mean_eig"]
std_eig       = data["std_eig"]
mean_eig_init = data["mean_eig_init"]
std_eig_init  = data["std_eig_init"]
total_param_arr = data["total_param_arr"]
peak_pos      = data["peak_pos"]

# ── Retrain one model per depth (same seed=43) to get trained weights ─
# We use rep 0 / seed 43 to get a representative trained model.
# If you saved model weights in your pkl you can load them instead.

seed = 43
weights = __import__('numpy').ones((n_qubits, n_qubits))
X_train, y_train, X_test, y_test = load_synthetic_multiclass_linear(
    weights, input_dim=n_qubits, n_train=n_train,
    output_dim=K, n_test=1000, noise_std=0.5, seed=seed
)

# Use a small test subset for speed (50 samples is enough for a sanity check;
# swap X_test_small -> X_test for the full bound)
X_test_small = X_test[:50]
y_test_small = y_test[:50]

lower_bounds = []
#single depth just for testing
for depth in [depths[0]]:
    torch.manual_seed(seed)
    print(f"\nDepth {depth} — retraining to get weights...")

    qlayer = create_quantum_layer(n_qubits=n_qubits, depth=depth, n_classes=K)
    model  = QuantumModel(q_layer=qlayer, sm_temp=0.1, c=c)

    # Quick retrain (same hyperparams as your main script)
    n_params = sum(p.numel() for p in model.parameters())
    lr = 1.0 / n_params
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    #set range to 1600
    for _ in range(100):
        opt.zero_grad()
        loss_fn(model(X_train), y_train).backward()
        opt.step()

    lb = compute_theorem8_lower_bound(
        model, X_train, y_train, X_test_small, y_test_small
    )
    lower_bounds.append(lb["lower_bound"])

    verdict = "✓" if lb["lower_bound"] <= nn.MSELoss()(model(X_test), y_test).item() + 1e-9 else "✗ VIOLATED"
    print(f"  p={n_params:4d} | lb={lb['lower_bound']:.4f} | "
          f"σ²_min={lb['sigma2_min']:.2e} | "
          f"λ_min(Cf)={lb['lambda_min_Cf']:.2e} | "
          f"λ_r(H)={lb['lambda_r_H']:.2e} | {verdict}")

lower_bounds = np.array(lower_bounds)

# ── Plot ──────────────────────────────────────────────────────────────

fig, ax1 = plt.subplots(figsize=(9, 5))

# Population loss (from saved results)
ax1.plot(params, mean_losses, marker='o', color='tab:red', label="Population loss")
ax1.fill_between(params, mean_losses - std_losses, mean_losses + std_losses,
                 alpha=0.2, color='tab:red')

# Lower bound (new)
ax1.plot(total_param_arr, lower_bounds, marker='s', color='tab:green',
         linestyle='--', lw=1.8, label="Lower bound (Thm 8)")

ax1.set_xlabel("Number of Parameters", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.axvline(x=peak_pos, color='gray', linestyle='--',
            label=f'Interpolation threshold (K·N = {peak_pos})')
ax1.legend(fontsize=11, loc='upper right')

# Min eigenvalue on right axis (from saved results, unchanged)
ax2 = ax1.twinx()
ax2.plot(total_param_arr, mean_eig, marker='o', color='tab:blue', label='λ_min (trained)')
ax2.fill_between(total_param_arr,
                 np.array(mean_eig) - np.array(std_eig),
                 np.array(mean_eig) + np.array(std_eig),
                 alpha=0.2, color='tab:blue')
ax2.plot(total_param_arr, mean_eig_init, marker='o', color='lightblue', label='λ_min (init)')
ax2.set_yscale('log')
ax2.set_ylabel("Min. non-zero eigenvalue (log)", fontsize=12)
ax2.legend(fontsize=11, loc='upper left')

plt.grid(True)
plt.tight_layout()
plt.savefig("lower_bound_plot.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → lower_bound_plot.png")
plt.show()
