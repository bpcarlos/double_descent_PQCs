import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
from matplotlib.lines import Line2D 

# --- Main Script ---

# 1. Define experiment structure
datasets = ['MNIST-1D', 'Fashion MNIST', 'Regression']

# Map datasets to their specific training set sizes (6 sizes per dataset)
train_sizes_map = {
    'MNIST-1D':      ['21', '30', '39', '48'],
    'Fashion MNIST': ['21', '30', '39', '48'],
    'Regression':    ['21', '30', '39', '48']
}

# 2. Setup plot styling
# CHANGE: Expanded to 6 colorblind-friendly colors
colors = {
    'red': '#d55e00',
    'sky_blue': '#56b4e9',
    'green': '#009e73',
    'yellow': '#d6cb3b',
    'dark_blue': '#0072b2',
    'purple': '#cc79a7'
}
plot_colors = [
    colors['red'], colors['sky_blue'], colors['green'], 
    colors['yellow'], colors['dark_blue'], colors['purple']
]

# CHANGE: Added 3 new markers (square, diamond, down-triangle)
plot_markers = ['o', 'x', '^', 's', 'D', 'v']

# Configure plot style
fontfig = 12
plt.rcParams.update({'font.size': fontfig, 'font.family': 'times'})

with plt.style.context(['science']):
    plt.rcParams['axes.linewidth'] = 1.1

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(14/1.15, 7/1.15), sharex=True)
    
    fig.subplots_adjust(wspace=0)

    # 3. Loop through each DATASET (columns)
    for j, dataset_name in enumerate(datasets):
        ax_risk = axes[0, j]
        ax_eig = axes[1, j]

        # Set the title for the column
        ax_risk.set_title(dataset_name, fontsize=16)
        
        # Get the specific sizes for this dataset
        current_sizes = train_sizes_map[dataset_name]

        # Loop through each TRAINING SIZE (curves)
        for i, size_str in enumerate(current_sizes):
            # Construct dynamic label
            current_label = f"N={size_str}"

            # Assuming the data generation script has been run and folders exist
            file_path = os.path.join(dataset_name, size_str, "experiment_results.pkl")
            
            # Simple error handling in case data hasn't been generated yet
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            with open(file_path, "rb") as f:
                data = pickle.load(f)
            
            # CHANGE: Slices [1:] removed to plot all data points
            params = np.array(data["params"])
            mean_losses = np.array(data["mean_losses"])
            std_losses = np.array(data["std_losses"])
            mean_eig = np.array(data["mean_eig"])
            std_eig = np.array(data["std_eig"])
            mean_eig_init = np.array(data["mean_eig_init"])
            std_eig_init = np.array(data["std_eig_init"])
            
            # Peak pos is likely a scalar value for a vertical line, so we leave it as is
            peak_pos = data["peak_pos"]

            # --- Plot on TOP ROW: Population Risk ---
            ax_risk.plot(params, mean_losses, marker=plot_markers[i], color=plot_colors[i], label=current_label, markersize=4, linewidth=1.5)
            ax_risk.fill_between(params, mean_losses - std_losses, mean_losses + std_losses, alpha=0.15, color=plot_colors[i])
            ax_risk.axvline(x=peak_pos, color=plot_colors[i], linestyle='--', linewidth=1.3, alpha=0.8, zorder=1)

            # --- Plot on BOTTOM ROW: Eigenvalues ---
            ax_eig.plot(params, mean_eig, marker=plot_markers[i], color=plot_colors[i], markersize=4, linewidth=1.5)
            ax_eig.fill_between(params, mean_eig - std_eig, mean_eig + std_eig, alpha=0.15, color=plot_colors[i])
            
            ax_eig.plot(params, mean_eig_init, marker=plot_markers[i], color=plot_colors[i], markersize=4, linestyle=':', linewidth=1.5)
            ax_eig.fill_between(params, mean_eig_init - std_eig_init, mean_eig_init + std_eig_init, alpha=0.1, color=plot_colors[i])
            ax_eig.axvline(x=peak_pos, color=plot_colors[i], linestyle='--', linewidth=1.3, alpha=0.8, zorder=1)

    # 4. Finalize plot details and styling (outside the main loop)
    subplot_letters = ['(a)', '(b)', '(c)']
    for i in range(2): # Rows
        for j in range(3): # Columns
            ax = axes[i, j]
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.minorticks_on()
            ax.grid(False)
            
            if i == 0: # Top Row (Risk)
                ax.text(-0.1, 1.05, subplot_letters[j], transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
                
                # Add legend ONLY to the first column in the top row, in the bottom right corner
                if j == 0:
                    legend = ax.legend(fontsize=10, frameon=True, fancybox=False, loc='lower right')
                    legend.get_frame().set_edgecolor('black')
                    legend.get_frame().set_linewidth(1.0)

                # Set axis label only on the first column
                if j == 0:
                    ax.set_ylabel("Test loss")

            if i == 1: # Bottom Row (Eigenvalues)
                ax.set_xlabel("Number of parameters")
                ax.set_yscale('log')
                
                # Set axis label and custom legend only on the first column
                if j == 0:
                    ax.set_ylabel("Min. non-zero eigenvalue")
                    
                    # Create a custom legend for line styles
                    legend_elements = [
                        Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='After training'),
                        Line2D([0], [0], color='black', linestyle=':', linewidth=1.5, label='Initialization')
                    ]
                    legend = ax.legend(handles=legend_elements, fontsize=10, frameon=True, fancybox=False, loc='upper left')
                    legend.get_frame().set_edgecolor('black')
                    legend.get_frame().set_linewidth(1.0)
    
    # Adjust layout and display the plot
    fig.tight_layout(pad=1.5, rect=[0, 0, 1, 0.96])
    fig.savefig('final_grid_plot.pdf', bbox_inches='tight')
    plt.show()
    
# =============================================================================
# NEW SNIPPET: Figure 2 - Training Loss and Jacobian Rank
# Append this directly to the bottom of your previous script!
# =============================================================================

with plt.style.context(['science']):
    # Create a new 2x3 grid for the new metrics
    fig2, axes2 = plt.subplots(2, 3, figsize=(14/1.15, 7/1.15), sharex=True)
    fig2.subplots_adjust(wspace=0)

    # Loop through each DATASET (columns)
    for j, dataset_name in enumerate(datasets):
        ax_train = axes2[0, j]
        ax_rank = axes2[1, j]

        # Set the title for the column
        ax_train.set_title(dataset_name, fontsize=16)
        
        # Get the specific sizes for this dataset
        current_sizes = train_sizes_map[dataset_name]

        # Loop through each TRAINING SIZE (curves)
        for i, size_str in enumerate(current_sizes):
            current_label = f"N={size_str}"

            file_path = os.path.join(dataset_name, size_str, "experiment_results.pkl")
            
            if not os.path.exists(file_path):
                continue

            with open(file_path, "rb") as f:
                data = pickle.load(f)
            
            params = np.array(data["params"])
            peak_pos = data["peak_pos"]

            # Extract Training Loss
            results_arr = np.array(data["results"])
            reps = results_arr.shape[0]
            mean_train_loss = np.mean(results_arr[:, :, 2], axis=0)
            std_train_loss = np.std(results_arr[:, :, 2], axis=0) / np.sqrt(reps)
            
            # Extract Jacobian Rank (from the last repetition saved in the dict)
            rank_arr = np.array(data["rank_arr"])

            # --- Plot TOP ROW: Training Loss ---
            ax_train.plot(params, mean_train_loss, marker=plot_markers[i], color=plot_colors[i], label=current_label, markersize=4, linewidth=1.5)
            ax_train.fill_between(params, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.15, color=plot_colors[i])
            ax_train.axvline(x=peak_pos, color=plot_colors[i], linestyle='--', linewidth=1.3, alpha=0.8, zorder=1)

            # --- Plot BOTTOM ROW: Jacobian Rank ---
            ax_rank.plot(params, rank_arr, marker=plot_markers[i], color=plot_colors[i], markersize=4, linewidth=1.5)
            ax_rank.axvline(x=peak_pos, color=plot_colors[i], linestyle='--', linewidth=1.3, alpha=0.8, zorder=1)

    # Finalize Figure 2 styling
    for i in range(2): # Rows
        for j in range(3): # Columns
            ax = axes2[i, j]
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.minorticks_on()
            ax.grid(False)
            
            if i == 0: # Top Row (Train Loss)
                ax.text(-0.1, 1.05, subplot_letters[j], transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
                # ax.set_yscale('log')
                
                # Added the identical condition here for the second figure
                if j == 0:
                    legend = ax.legend(fontsize=10, frameon=True, fancybox=False, loc='upper right')
                    legend.get_frame().set_edgecolor('black')
                    legend.get_frame().set_linewidth(1.0)

                if j == 0:
                    ax.set_ylabel("Training loss")

            if i == 1: # Bottom Row (Rank)
                ax.set_xlabel("Number of parameters")
                
                if j == 0:
                    ax.set_ylabel("Jacobian Rank")

    # Adjust layout and save the second plot
    fig2.tight_layout(pad=1.5, rect=[0, 0, 1, 0.96])
    fig2.savefig('final_grid_plot_train_rank.pdf', bbox_inches='tight')
    plt.show()