# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 10:21:47 2026

@author: jpauya1
"""


# ============================================================
# Validation: PINN vs Analytical (same chi grid)
# Output folder: runs_validation_PINN_base_case
#
# Requirements implemented:
# (1) Every 1000 epochs: save snapshot plot (PINN + Okwen only) per pair
# (2) Final per-pair plot: 3 subplots:
#     - Top-left: PINN + Analytical + Okwen
#     - Top-right: Loss history (log)
#     - Bottom-center: Error history (L2 vs epoch) between PINN and Analytical
# ============================================================

# ============================================================
# Validation: PINN vs Analytical (same chi grid)
# Output folder: runs_validation_PINN_base_case
#

# Requirements implemented:
# (1) Every 1000 epochs: save snapshot plot (PINN + Okwen only) per pair
# (2) Final per-pair plot: 3 subplots:
#     - Top-left: PINN + Analytical + Okwen
#     - Top-right: Loss history (log)
#     - Bottom-center: Error history (L2 vs epoch) between PINN and Analytical
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import shutil
import itertools
import gc
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ============================================================
# Global Random Seed for Reproducibility
# ============================================================
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across PyTorch, NumPy, and CUDA"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Okwen correlation (KEEP)
# ============================================================
def compute_chi_max(M, Gamma):
    return (0.0324 * M - 0.0952) * Gamma + (0.1778 * M + 5.9682) * (Gamma ** 0.5) + 1.6962 * M - 3.0472


def is_okwen_valid(M, Gamma):
    """
    Check if M and Gamma values are within valid range for Okwen correlation.
    Based on correlation range: 5.0 < M < 20.0, 0.5 < Gamma < 50.0
    """
    return (5.0 < M < 20.0) and (0.5 < Gamma < 50.0)


# ============================================================
# Analytical solution (piecewise) - evaluate on SAME chi grid
# ============================================================
def h_analytical(chi, M):
    """
    Piecewise analytical solution (Gamma does NOT appear in this closed form):
      h = 0                                for chi <= 2/M
      h = (M - sqrt(2M/chi)) / (M - 1)     for 2/M < chi < 2M
      h = 1                                for chi >= 2M
    """
    chi = np.asarray(chi, dtype=float)
    h = np.zeros_like(chi)

    left = chi <= (2.0 / M)
    mid = (chi > (2.0 / M)) & (chi < (2.0 * M))
    right = chi >= (2.0 * M)

    h[left] = 0.0
    h[mid] = (M - np.sqrt(2.0 * M / chi[mid])) / (M - 1.0)
    h[right] = 1.0
    return h


# ============================================================
# PINN model (same style, bounded output)
# ============================================================
class PINN(nn.Module):
    def __init__(self, layers, activation_function, init_seed=None):
        super().__init__()
        activation_map = {
            "tanh": nn.Tanh()
        }
        if activation_function not in activation_map:
            raise ValueError(f"Unknown activation_function: {activation_function}")

        activation = activation_map[activation_function]

        modules = []
        modules.append(nn.Linear(1, layers[0]))
        modules.append(activation)

        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            modules.append(activation)

        modules.append(nn.Linear(layers[-1], 1))
        modules.append(nn.Sigmoid())  # force 0..1 for h_aD

        self.layers = nn.Sequential(*modules)

        if init_seed is not None:
            self._init_weights(init_seed)

    def _init_weights(self, seed):
        torch.manual_seed(seed)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, chi):
        return self.layers(chi)


# ============================================================
# Solver class
# ============================================================
class PINNSolver:
    def __init__(self, M_values, Gamma_values, layers, activation_function, optimizer_name, output_dir, chi_min=0.0, N=100000, epochs=50000, learning_rate=1e-5, tol=1e-5, init_seed=42, snapshot_every=1000, error_eval_points=5000, additional_epochs_after_tolerance=5000):
        super().__init__()

        self.layers = layers
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.N = N
        self.epochs = epochs
        self.snapshot_every = snapshot_every
        self.error_eval_points = error_eval_points
        self.additional_epochs_after_tolerance = additional_epochs_after_tolerance

        self.devices = [torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")]

        self.M_values = M_values
        self.Gamma_values = Gamma_values
        self.chi_min = chi_min
        self.tol = tol
        self.output_dir = output_dir

        # Create pair-specific directories and store them in lists
        self.snap_dirs = []
        self.pair_dirs = []
        
        for idx, (M, Gamma) in enumerate(zip(M_values, Gamma_values)):
            pair_folder = os.path.join(self.output_dir, f"pair_{idx+1}_M{M}_Gamma{Gamma}")
            
            # Create main pair folder and snapshots subfolder
            if not os.path.exists(pair_folder):
                os.makedirs(pair_folder)
            snapshots_folder = os.path.join(pair_folder, "snapshots")
            os.makedirs(snapshots_folder, exist_ok=True)

            # Store directories for each pair
            self.snap_dirs.append(snapshots_folder)
            self.pair_dirs.append(pair_folder)
            self.models = {device: [] for device in self.devices}
            self.optimizers = {device: [] for device in self.devices}
            self.chi_max_values = {device: [] for device in self.devices}
            self.chi_grids = {device: [] for device in self.devices}

        self.loss_history = {device: [[] for _ in M_values] for device in self.devices}
        self.convergence_epochs = {device: [None for _ in M_values] for device in self.devices}

        # (NEW) error history: store L2 error between PINN and analytical per epoch per pair
        self.error_history = {device: [[] for _ in M_values] for device in self.devices}
        
        # (NEW) individual loss component histories
        self.ode_loss_history = {device: [[] for _ in M_values] for device in self.devices}
        self.boundary_loss_history = {device: [[] for _ in M_values] for device in self.devices}
        self.integral_loss_history = {device: [[] for _ in M_values] for device in self.devices}

        # Early stopping state variables
        self.tolerance_reached_epochs = {device: [None for _ in M_values] for device in self.devices}
        self.early_stop_epochs = {device: [None for _ in M_values] for device in self.devices}

        self.training_times = {device: [0.0] * len(M_values) for device in self.devices}
        self.device_times = {device: 0.0 for device in self.devices}

        for device in self.devices:
            for M, Gamma in zip(M_values, Gamma_values):
                model = PINN(self.layers, self.activation_function, init_seed=init_seed).to(device)
                optimizer = self.get_optimizer(model)

                chi_max = 2.0 * M
                chi = torch.linspace(chi_min, chi_max, N).view(-1, 1).to(device)

                self.models[device].append(model)
                self.optimizers[device].append(optimizer)
                self.chi_max_values[device].append(chi_max)
                self.chi_grids[device].append(chi)

                print(fr"[INIT] Device: {device}, M={M}, Γ={Gamma}, χ_max0={chi_max}")

    def get_optimizer(self, model):
        if self.optimizer_name == "Adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "SGD":
            return optim.SGD(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def loss_function(self, chi, model, M, Gamma, chi_max):
        chi = chi.requires_grad_(True).float()
        h_aD = model(chi)
        h_aD_chi = torch.autograd.grad(h_aD, chi, grad_outputs=torch.ones_like(h_aD), create_graph=True)[0]

        term1 = -chi * h_aD_chi
        F_h_aD = h_aD / (h_aD + M * (1 - h_aD))
        G_h_aD = M * h_aD * (1 - h_aD) / (h_aD + M * (1 - h_aD))
        P = F_h_aD - G_h_aD * 2 * Gamma * chi * h_aD_chi
        P_chi = torch.autograd.grad(P, chi, grad_outputs=torch.ones_like(P), create_graph=True)[0]
        term2 = 2 * P_chi
        ODE_loss = torch.mean((term1 + term2) ** 2)

        chi_min_tensor = torch.tensor([[self.chi_min]], dtype=torch.float, device=chi.device)
        chi_max_tensor = torch.tensor([[chi_max]], dtype=torch.float, device=chi.device)
        h_aD_min = model(chi_min_tensor)
        h_aD_max = model(chi_max_tensor)
        boundary_loss = (h_aD_min - 0) ** 2 + (h_aD_max - 1) ** 2

        integral_values = 1 - h_aD
        integral_result = torch.trapz(integral_values.squeeze(), chi.squeeze())
        integral_loss = torch.mean((integral_result - 2) ** 2)

        total_loss = ODE_loss + boundary_loss + integral_loss
        return total_loss, integral_result, ODE_loss, boundary_loss, integral_loss

    def add_plot_annotations(self, ax):
        """Add plot annotations with automatic best location"""
        params_text = (
            f"Layers: {self.layers}\n"
            f"Activation: {self.activation_function}\n"
            f"Learning Rate: {self.learning_rate:.0e}\n"
            f"Optimizer: {self.optimizer_name}"
        )
        
        # Create a dummy plot for legend to use loc='best'
        ax.plot([], [], ' ', label=params_text)
        legend = ax.legend(
            loc='best',
            frameon=True,
            fancybox=True,
            shadow=False,
            facecolor='white',
            edgecolor='black',
            fontsize=10,
            borderpad=0.5,  # Equal padding around text
            labelspacing=0.3,  # Spacing between lines
            handlelength=0,  # No space for legend handles
            handletextpad=0,  # No space between handle and text
            borderaxespad=0.4  # Padding between legend and axes
        )
        legend.get_frame().set_alpha(0.9)

    # ============================================================
    # (1) Snapshot plot: PINN + Okwen ONLY (no analytical)
    # saved every snapshot_every epochs per pair
    # ============================================================
    def save_snapshot_plot(self, device, epoch, pair_index):
        M = self.M_values[pair_index]
        Gamma = self.Gamma_values[pair_index]
    
        chi_t = self.chi_grids[device][pair_index]
        chi_np = chi_t.detach().cpu().numpy().squeeze()
    
        with torch.no_grad():
            h_pinn = self.models[device][pair_index](chi_t).detach().cpu().numpy().squeeze()
    
        # Collect the loss histories for all loss components
        physics_loss = np.array(self.ode_loss_history[device][pair_index], dtype=float)
        boundary_loss = np.array(self.boundary_loss_history[device][pair_index], dtype=float)
        integral_loss = np.array(self.integral_loss_history[device][pair_index], dtype=float)
        
        # Calculate total loss history and get current total loss
        total_loss = physics_loss + boundary_loss + integral_loss
        current_total_loss = total_loss[-1] if len(total_loss) > 0 else float('nan')
        
        # Also compute RMSE for reference (but don't display in main title)
        h_ana = h_analytical(chi_np, M)
        rmse = float(np.sqrt(np.mean((h_pinn - h_ana) ** 2)))
    
        # Create a 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Final Solution | M={M}, Γ={Gamma} | Total Loss: {current_total_loss:.2e}", fontsize=14)
    
        # --- Top-left: PINN + Okwen
        ax1 = axes[0, 0]
        ax1.plot(chi_np, h_pinn, color="blue", linewidth=2.2, label=f"PINN ($M={M}$, $\Gamma={Gamma}$)")
        
        # Only plot Okwen point if within valid range
        if is_okwen_valid(M, Gamma):
            okwen_point = compute_chi_max(M, Gamma)
            ax1.plot(okwen_point, 1, 'o', color="brown", markersize=8, markerfacecolor="green", label="Okwen")
        
        ax1.set_xlabel(r"$\chi$", fontsize=13)
        ax1.set_ylabel(r"$h_{aD}(\chi)$", fontsize=13)
        ax1.set_title("PINN Solution", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
        # --- Top-right: Physics Error (Residual and BC separately)
        ax2 = axes[0, 1]
        epochs_array = np.arange(1, len(physics_loss) + 1)
        ax2.plot(epochs_array, physics_loss, color="red", linewidth=2.0, label="Residual")
        ax2.plot(epochs_array, boundary_loss, color="orange", linewidth=2.0, label="Boundary Conditions")
        ax2.set_yscale("log")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Local Physics Loss", fontsize=12)
        ax2.set_title("Local Physics Loss Components", fontsize=14)
        ax2.legend()
        ax2.grid(True, which="both", alpha=0.3)
    
        # --- Bottom-left: Integral Error
        ax3 = axes[1, 0]
        ax3.plot(epochs_array, integral_loss, color="green", linewidth=2.0)
        ax3.set_yscale("log")
        ax3.set_xlabel("Epoch", fontsize=12)
        ax3.set_ylabel("Integral Loss", fontsize=12)
        ax3.set_title(r"Integral Loss ($\int (1-h)\, d\chi = 2$)", fontsize=14)
        ax3.grid(True, which="both", alpha=0.3)
    
        # --- Bottom-right: Total Loss
        ax4 = axes[1, 1]
        ax4.plot(epochs_array, total_loss, color="violet", linewidth=2.0)
        ax4.set_yscale("log")
        ax4.set_xlabel("Epoch", fontsize=12)
        ax4.set_ylabel("Total Loss", fontsize=12)
        ax4.set_title("Total Loss (Sum of All Losses)", fontsize=14)
        ax4.grid(True, which="both", alpha=0.3)
    
        # Adjust layout for the title
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
        # Save to snapshots folder for the current pair
        snapshot_path = os.path.join(self.snap_dirs[pair_index], f"pair_{pair_index+1}_M_{M}_Gamma_{Gamma}_epoch_{epoch}_{device}_snapshot.png")
        plt.savefig(snapshot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    
        # For the final plot (outside snapshots folder but inside the pair folder)
        if epoch == self.epochs:
            final_plot_path = os.path.join(self.pair_dirs[pair_index], f"pair_{pair_index+1}_M_{M}_Gamma_{Gamma}_final_{device}.png")
            fig.savefig(final_plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


    # ============================================================
    # Compute error metric (L2) between PINN and analytical
    # on SAME chi grid at current epoch (subsample for speed)
    # ============================================================
    def compute_epoch_error_L2(self, device, pair_index):
        M = self.M_values[pair_index]

        chi_t_full = self.chi_grids[device][pair_index]
        chi_np_full = chi_t_full.detach().cpu().numpy().squeeze()

        # subsample for speed (uniform)
        if chi_np_full.size > self.error_eval_points:
            idx = np.linspace(0, chi_np_full.size - 1, self.error_eval_points).astype(int)
            chi_np = chi_np_full[idx]
            chi_t = chi_t_full[idx].view(-1, 1)
        else:
            chi_np = chi_np_full
            chi_t = chi_t_full

        with torch.no_grad():
            h_pinn = self.models[device][pair_index](chi_t).detach().cpu().numpy().squeeze()

        h_ana = h_analytical(chi_np, M)
        err = h_pinn - h_ana
        L2 = float(np.sqrt(np.mean(err ** 2)))
        return L2

    # ============================================================
    # Plot individual loss components alongside total loss
    # ============================================================
    def plot_loss_components(self, device, pair_index):
        M = self.M_values[pair_index]
        Gamma = self.Gamma_values[pair_index]

        # Get loss histories
        total_loss = np.array(self.loss_history[device][pair_index], dtype=float)
        ode_loss = np.array(self.ode_loss_history[device][pair_index], dtype=float)
        boundary_loss = np.array(self.boundary_loss_history[device][pair_index], dtype=float)
        integral_loss = np.array(self.integral_loss_history[device][pair_index], dtype=float)

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        #fig.suptitle(fr"Final Epoch | Loss Components | $M={M}$, $\Gamma={Gamma}$ ({device})", fontsize=16)

        epochs = np.arange(1, len(total_loss) + 1)

        # ODE loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, ode_loss, color="red", linewidth=2.0)
        ax1.set_yscale("log")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Residual Loss", fontsize=12)
        ax1.set_title("Residual Loss", fontsize=14)
        ax1.grid(True, which="both", alpha=0.3)

        # Boundary loss
        ax2 = axes[0, 1]
        ax2.plot(epochs, boundary_loss, color="orange", linewidth=2.0)
        ax2.set_yscale("log")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Boundary Loss", fontsize=12)
        ax2.set_title("Boundary Loss (h=0, h=1)", fontsize=14)
        ax2.grid(True, which="both", alpha=0.3)

        # Integral loss
        ax3 = axes[1, 0]
        ax3.plot(epochs, integral_loss, color="green", linewidth=2.0)
        ax3.set_yscale("log")
        ax3.set_xlabel("Epoch", fontsize=12)
        ax3.set_ylabel("Integral Loss", fontsize=12)
        ax3.set_title(r"Integral Loss ($\int (1-h)\, d\chi = 2$)", fontsize=14)
        ax3.grid(True, which="both", alpha=0.3)

        # Total loss
        ax4 = axes[1, 1]
        ax4.plot(epochs, total_loss, color="violet", linewidth=2.0)
        ax4.set_yscale("log")
        ax4.set_xlabel("Epoch", fontsize=12)
        ax4.set_ylabel("Total Loss", fontsize=12)
        ax4.set_title("Total Loss", fontsize=14)
        ax4.grid(True, which="both", alpha=0.3)
        
        plt.tight_layout()
        out_path = os.path.join(self.pair_dirs[pair_index], f"LOSS_COMPONENTS_pair_{pair_index+1}_M_{M}_Gamma_{Gamma}_{device}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()


    # ============================================================
    # (2) Final per-pair plot: 3 subplots
    # Top-left: solution (PINN + analytical + Okwen)
    # Top-right: loss history for this pair (log)
    # Bottom-center: error history (L2) for this pair
    # ============================================================
    def plot_final_pair_3panel(self, device, pair_index):
        M = self.M_values[pair_index]
        Gamma = self.Gamma_values[pair_index]

        chi_t = self.chi_grids[device][pair_index]
        chi_np = chi_t.detach().cpu().numpy().squeeze()

        with torch.no_grad():
            h_pinn = self.models[device][pair_index](chi_t).detach().cpu().numpy().squeeze()

        h_ana = h_analytical(chi_np, M)
        okwen_point = compute_chi_max(M, Gamma)

        loss_hist = np.array(self.loss_history[device][pair_index], dtype=float)
        err_hist = np.array(self.error_history[device][pair_index], dtype=float)

        # Layout: 2 columns on top, 1 centered bottom
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        ax_sol = fig.add_subplot(gs[0, 0])
        ax_loss = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[1, :])  # span both columns (centered bottom)

        # --- top-left solution
        ax_sol.plot(chi_np, h_pinn, color="blue", linewidth=2.5, label="PINN")
        ax_sol.plot(chi_np, h_ana, color="red", linewidth=2.5, label="Analytical")
        
        # Only plot Okwen point if within valid range
        if is_okwen_valid(M, Gamma):
            okwen_point = compute_chi_max(M, Gamma)
            ax_sol.plot(okwen_point, 1, 'o', color="brown", markersize=9, markerfacecolor="green", label="Okwen")
        
        ax_sol.set_xlabel(r"$\chi$", fontsize=16)
        ax_sol.set_ylabel(r"$h_{aD}(\chi)$", fontsize=16)
        ax_sol.set_title(fr"PINN Solution | $M={M}$, $\Gamma={Gamma}$", fontsize=18)
        ax_sol.grid(True, alpha=0.3)
        ax_sol.legend(loc="lower right", fontsize=14)

        # --- top-right loss
        ax_loss.plot(loss_hist, color="violet", linewidth=2.0)
        ax_loss.set_yscale("log")
        ax_loss.set_xlabel("Epoch", fontsize=16)
        ax_loss.set_ylabel("Total Loss", fontsize=16)
        ax_loss.set_title("Total Loss", fontsize=18)
        ax_loss.grid(True, which="both", alpha=0.3)
        self.add_plot_annotations(ax_loss)

        # --- bottom error history
        ax_err.plot(err_hist, color="green", linewidth=2.0)
        ax_err.set_xlabel("Epoch", fontsize=16)
        ax_err.set_ylabel(r"$L_2(h_{\mathrm{PINN}}-h_{\mathrm{analytical}})$", fontsize=16)
        ax_err.set_title("Error History (Analytical vs PINN)", fontsize=18)
        ax_err.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(self.pair_dirs[pair_index], f"FINAL_3PANEL_pair_{pair_index+1}_M_{M}_Gamma_{Gamma}_{device}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()
        

    def save_loss_histories_to_json(self):
        """Save loss histories to JSON files for comparison"""
        import json
        
        for device in self.devices:
            for pair_idx in range(len(self.M_values)):
                M = self.M_values[pair_idx]
                Gamma = self.Gamma_values[pair_idx]
                
                # Create pair directory name
                pair_dir_name = f"pair_{pair_idx + 1}_M{M}_Gamma{Gamma}"
                pair_dir = os.path.join(self.output_dir, pair_dir_name)
                
                # Ensure pair directory exists
                os.makedirs(pair_dir, exist_ok=True)
                
                # Prepare loss data
                loss_data = {
                    'total_loss': [float(loss) for loss in self.loss_history[device][pair_idx]],
                    'ode_loss': [float(loss) for loss in self.ode_loss_history[device][pair_idx]],
                    'boundary_loss': [float(loss) for loss in self.boundary_loss_history[device][pair_idx]],
                    'integral_loss': [float(loss) for loss in self.integral_loss_history[device][pair_idx]],
                    'error_history': [float(err) for err in self.error_history[device][pair_idx]],
                    'epochs': list(range(len(self.loss_history[device][pair_idx])))
                }
                
                # Save to JSON file
                json_file = os.path.join(pair_dir, 'loss_histories.json')
                with open(json_file, 'w') as f:
                    json.dump(loss_data, f, indent=2)
                
                print(f"Saved loss histories to {json_file}")
                
                # Save final PINN solution
                model = self.models[device][pair_idx]
                chi = self.chi_grids[device][pair_idx]
                
                with torch.no_grad():
                    h_pinn = model(chi).detach().cpu().numpy().squeeze()
                
                solution_data = {
                    'chi': chi.detach().cpu().numpy().squeeze().tolist(),
                    'h_pinn': h_pinn.tolist()
                }
                
                solution_file = os.path.join(pair_dir, 'solution_data.json')
                with open(solution_file, 'w') as f:
                    json.dump(solution_data, f, indent=2)
                
                print(f"Saved PINN solution to {solution_file}")

    def save_summary(self):
        summary_filename = os.path.join(self.output_dir, "training_summary_validation.txt")
        with open(summary_filename, "w", encoding='utf-8') as f:
            f.write("\n============================== SUMMARY REPORT ==============================\n")
            f.write("| Device |   M   |  Γ  |  χ_min  |  χ_max  | χ_max(Okwen) | Integral | Final Loss | Final Err(L2) | Epochs | Time(s) | Tol Reached | Early Stop |\n")
            f.write("|--------|-------|-----|---------|---------|-------------|----------|-----------|--------------|--------|---------|-------------|------------|\n")

            for device in self.devices:
                for i, (M, Gamma) in enumerate(zip(self.M_values, self.Gamma_values)):
                    model = self.models[device][i]
                    chi = self.chi_grids[device][i]
                    chi_max = self.chi_max_values[device][i]
                    chi_max_ok = compute_chi_max(M, Gamma)

                    with torch.no_grad():
                        integral_values = 1 - model(chi)
                        integral_val = torch.trapz(integral_values.squeeze(), chi.squeeze()).item()

                    final_loss = self.loss_history[device][i][-1] if len(self.loss_history[device][i]) else float('nan')
                    final_err = self.error_history[device][i][-1] if len(self.error_history[device][i]) else float('nan')
                    epochs_ran = len(self.loss_history[device][i])
                    tsec = self.training_times[device][i]
                    tol_reached = self.tolerance_reached_epochs[device][i] if self.tolerance_reached_epochs[device][i] is not None else "N/A"
                    early_stop = self.early_stop_epochs[device][i] if self.early_stop_epochs[device][i] is not None else "N/A"

                    f.write(f"| {str(device):6s} | {M:5.2f} | {Gamma:3.2f} | {self.chi_min:7.3f} | {chi_max:7.3f} |"
                            f" {chi_max_ok:11.3f} | {integral_val:8.4f} | {final_loss:9.2e} | {final_err:12.4e} |"
                            f" {epochs_ran:6d} | {tsec:7.2f} | {tol_reached:11s} | {early_stop:10s} |\n")

            f.write("==========================================================================\n")

    def train(self):
        for device in self.devices:
            start_time = time.time()
    
            # Training loop over epochs
            for epoch in range(1, self.epochs + 1):
                for i, (M, Gamma) in enumerate(zip(self.M_values, self.Gamma_values)):
                    pair_start = time.time()
    
                    # Fetch model, optimizer, and grid for the current pair
                    model = self.models[device][i]
                    optimizer = self.optimizers[device][i]
                    chi_max = self.chi_max_values[device][i]
                    chi = self.chi_grids[device][i]
    
                    # Zero gradients, compute loss, and backpropagate
                    optimizer.zero_grad()
                    loss, integral_result, ode_loss, boundary_loss, integral_loss = self.loss_function(chi, model, M, Gamma, chi_max)
                    loss.backward()
                    optimizer.step()
    
                    # Store loss and individual components
                    self.loss_history[device][i].append(loss.item())
                    self.ode_loss_history[device][i].append(ode_loss.item())
                    self.boundary_loss_history[device][i].append(boundary_loss.item())
                    self.integral_loss_history[device][i].append(integral_loss.item())
                    
                    # Early stopping logic: check if tolerance is reached
                    if self.tolerance_reached_epochs[device][i] is None and loss.item() <= self.tol:
                        # First time reaching tolerance
                        self.tolerance_reached_epochs[device][i] = epoch
                        self.early_stop_epochs[device][i] = epoch + self.additional_epochs_after_tolerance
                        print(f"[EARLY_STOP] {device} pair_{i+1} reached tolerance {self.tol:.0e} at epoch {epoch}. Will stop at epoch {self.early_stop_epochs[device][i]}")
    
                    # Dynamically update chi_max (based on your specific logic)
                    integral_error = (integral_result - 2.0).item()
                    if integral_error > 0:
                        self.chi_max_values[device][i] += 0.0001 * abs(integral_error)
                    else:
                        self.chi_max_values[device][i] -= 0.0001 * abs(integral_error)
    
                    # Ensure chi_max doesn't drop below a threshold
                    if self.chi_max_values[device][i] <= 1e-8:
                        self.chi_max_values[device][i] = 1e-8
    
                    # Update chi grid
                    chi = torch.linspace(self.chi_min, self.chi_max_values[device][i], self.N).view(-1, 1).to(device)
                    self.chi_grids[device][i] = chi
    
                    # Record error history (L2 error between PINN and analytical)
                    errL2 = self.compute_epoch_error_L2(device, i)
                    self.error_history[device][i].append(errL2)
    
                    # Timing for each pair's computation
                    self.training_times[device][i] += (time.time() - pair_start)
                
                # Check if all pairs have reached their early stop epochs
                all_reached_early_stop = True
                for i in range(len(self.M_values)):
                    if self.early_stop_epochs[device][i] is None or epoch < self.early_stop_epochs[device][i]:
                        all_reached_early_stop = False
                        break
                
                if all_reached_early_stop:
                    print(f"[EARLY_STOP] {device} All pairs have completed additional epochs. Ending training at epoch {epoch}")
                    break
    
                # Snapshots every 1000 epochs (for all pairs)
                if epoch == 1 or (self.snapshot_every and epoch % self.snapshot_every == 0):
                    print(f"[{device}] saving snapshots at epoch={epoch}")
                    for pair_idx in range(len(self.M_values)):
                        self.save_snapshot_plot(device, epoch, pair_idx)
    
            # Track total training time for each device
            self.device_times[device] = time.time() - start_time
            print(f"[DONE] Training time on {device}: {self.device_times[device]:.2f} s")
    
        # Save summary after training
        self.save_summary()
        
        # Save loss histories to JSON for comparison
        self.save_loss_histories_to_json()
    
        # Save final plots and loss components after training is complete
        for device in self.devices:
            for pair_idx in range(len(self.M_values)):
                # Save final 3-panel plot for each pair (outside of snapshots folder)
                self.plot_final_pair_3panel(device, pair_idx)
    
                # Save individual loss components plot for each pair (outside of snapshots folder)
                self.plot_loss_components(device, pair_idx)


# ============================================================
# Results aggregation (lightweight)
# ============================================================
def parse_run_dir(run_dir):
    """
    Expected format:
    run_N_10000_layers_[16,16,16]_act_tanh
    """
    parts = run_dir.split('_')
    N_value = int(parts[2])
    layers_str = parts[4]
    activation = parts[6]
    layers = eval(layers_str)
    return layers, activation

def create_summary_dataframe(main_output_dir):
    results = []
    for run_dir in os.listdir(main_output_dir):
        full_path = os.path.join(main_output_dir, run_dir)
        if not os.path.isdir(full_path):
            continue

        try:
            N_value, layers, activation = parse_run_dir(run_dir)
        except Exception:
            continue

        summary_path = os.path.join(full_path, "training_summary_validation.txt")
        if not os.path.exists(summary_path):
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("|") and ("cuda" in line or "cpu" in line) and "Device" not in line:
                cols = [c.strip() for c in line.split("|") if c.strip()]
                try:
                    device = cols[0]
                    M = float(cols[1])
                    Gamma = float(cols[2])
                    chi_max = float(cols[4])
                    chi_max_ok = float(cols[5])
                    integral_val = float(cols[6])
                    final_loss = float(cols[7])
                    final_err = float(cols[8])
                    epochs_ran = int(cols[9])
                    time_s = float(cols[10])

                    results.append({
                        "Directory": run_dir,
                        "N_collocation": N_value,
                        "Layers": str(layers),
                        "Activation": activation,
                        "Device": device,
                        "M": M,
                        "Gamma": Gamma,
                        "chi_max_PINN": chi_max,
                        "chi_max_Okwen": chi_max_ok,
                        "Integral": integral_val,
                        "Final Loss": final_loss,
                        "Final Err L2": final_err,
                        "Epochs": epochs_ran,
                        "Training Time (s)": time_s
                    })
                except Exception:
                    continue

    return pd.DataFrame(results)


def plot_N_sensitivity(main_output_dir, M_target, Gamma_target):
    """
    Create a 3-panel comparison plot for different collocation sizes N.

    Panel 1: Final PINN solution for each N + analytical solution
    Panel 2: Total loss history for each N
    Panel 3: Error history for each N

    Assumes each run folder contains:
      pair_1_M{M}_Gamma{Gamma}/loss_histories.json
      pair_1_M{M}_Gamma{Gamma}/solution_data.json
    """

    color_map = {
        10000: "blue",
        50000: "green",
        100000: "violet"
    }

    run_data = []

    for run_dir in os.listdir(main_output_dir):
        full_run_path = os.path.join(main_output_dir, run_dir)
        if not os.path.isdir(full_run_path):
            continue

        match = re.match(r"run_N_(\d+)_layers_(\[[^\]]+\])_act_(.+)", run_dir)
        if not match:
            continue

        N_value = int(match.group(1))

        pair_dir = os.path.join(
            full_run_path,
            f"pair_1_M{M_target}_Gamma{Gamma_target}"
        )

        loss_file = os.path.join(pair_dir, "loss_histories.json")
        sol_file = os.path.join(pair_dir, "solution_data.json")

        if not (os.path.exists(loss_file) and os.path.exists(sol_file)):
            print(f"[WARN] Missing saved JSON files for N={N_value} in {pair_dir}")
            continue

        with open(loss_file, "r") as f:
            loss_data = json.load(f)

        with open(sol_file, "r") as f:
            sol_data = json.load(f)

        chi = np.array(sol_data["chi"], dtype=float)
        h_pinn = np.array(sol_data["h_pinn"], dtype=float)
        total_loss = np.array(loss_data["total_loss"], dtype=float)
        error_history = np.array(loss_data["error_history"], dtype=float)

        run_data.append({
            "N": N_value,
            "chi": chi,
            "h_pinn": h_pinn,
            "total_loss": total_loss,
            "error_history": error_history
        })

    if not run_data:
        print("[WARNING] No run data found for N-comparison plot.")
        return

    run_data = sorted(run_data, key=lambda x: x["N"])

    # Analytical solution from first chi grid
    chi_ref = run_data[0]["chi"]
    h_ana = h_analytical(chi_ref, M_target)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax_sol = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, :])

    # -----------------------------
    # Panel 1: Solution comparison
    # -----------------------------
    for item in run_data:
        N_value = item["N"]
        color = color_map.get(N_value, None)
        ax_sol.plot(
            item["chi"], item["h_pinn"],
            color=color,
            linewidth=2.0,
            label=fr"PINN ($N={N_value}$)"
        )

    ax_sol.plot(
        chi_ref, h_ana,
        color="red",
        linewidth=2.2,
        linestyle="-",
        label="Analytical"
    )

    ax_sol.set_xlabel(r"$\chi$", fontsize=16)
    ax_sol.set_ylabel(r"$h_{aD}(\chi)$", fontsize=16)
    ax_sol.set_title(fr"PINN Solution | $M={M_target}$, $\Gamma={Gamma_target}$", fontsize=18)
    ax_sol.grid(True, alpha=0.3)
    ax_sol.legend(fontsize=14, loc="lower right")

    # -----------------------------
    # Panel 2: Total loss histories
    # -----------------------------
    for item in run_data:
        N_value = item["N"]
        color = color_map.get(N_value, None)
        epochs = np.arange(1, len(item["total_loss"]) + 1)

        ax_loss.plot(
            epochs, item["total_loss"],
            color=color,
            linewidth=2.0,
            label=fr"$N={N_value}$"
        )

    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch", fontsize=16)
    ax_loss.set_ylabel("Total Loss", fontsize=16)
    ax_loss.set_title("Total Loss History", fontsize=18)
    ax_loss.grid(True, which="both", alpha=0.3)
    ax_loss.legend(fontsize=14)

    # -----------------------------
    # Panel 3: Error histories
    # -----------------------------
    for item in run_data:
        N_value = item["N"]
        color = color_map.get(N_value, None)
        epochs = np.arange(1, len(item["error_history"]) + 1)

        ax_err.plot(
            epochs, item["error_history"],
            color=color,
            linewidth=2.0,
            label=fr"$N={N_value}$"
        )

    ax_err.set_xlabel("Epoch", fontsize=16)
    ax_err.set_ylabel(r"$L_2(h_{\mathrm{PINN}}-h_{\mathrm{analytical}})$", fontsize=16)
    ax_err.set_title("Error History (Analytical vs PINN)", fontsize=18)
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(fontsize=14)

    plt.tight_layout()
    out_path = os.path.join(
        main_output_dir,
        f"N_comparison_3panel_M{M_target}_Gamma{Gamma_target}.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[INFO] Saved N-comparison figure to: {out_path}")
    
    
# ============================================================
# MAIN: Validation Base Case
# ============================================================
if __name__ == "__main__":

    # Set random seeds for reproducibility
    set_random_seeds(42)

    main_output_dir = "runs_validation_PINN_base_case"

    # Two pairs
    test_M = [6.0]
    test_Gamma = [0.4]

    # Grid search
    layer_options = [[16, 16, 16]]
    activation_functions = ["tanh"]
    optimizer_name = "Adam"

    # Training hyperparams
    N_values = [10000, 50000, 100000]
    epochs = 50000
    lr = 1e-5
    tol = 1e-5
    INIT_SEED = 42

    snapshot_every = 1000
    error_eval_points = 5000

    # Create main output directory fresh
    if os.path.exists(main_output_dir):
        shutil.rmtree(main_output_dir)
    os.makedirs(main_output_dir, exist_ok=True)

    best_loss = float("inf")
    best_config = None
    best_run_dir = None

    for N in N_values:
        for layers, activation in itertools.product(layer_options, activation_functions):
    
            layers_str = str(layers).replace(" ", "")
            run_dir = f"run_N_{N}_layers_{layers_str}_act_{activation}"
            full_run_dir = os.path.join(main_output_dir, run_dir)
    
            print("\n====================================================")
            print(f"RUN: N={N}, layers={layers}, activation={activation}")
            print("====================================================")
    
            try:
                solver = PINNSolver(
                    init_seed=INIT_SEED,
                    M_values=test_M,
                    Gamma_values=test_Gamma,
                    layers=layers,
                    activation_function=activation,
                    optimizer_name=optimizer_name,
                    output_dir=full_run_dir,
                    chi_min=0.0,
                    N=N,
                    epochs=epochs,
                    learning_rate=lr,
                    tol=tol,
                    snapshot_every=snapshot_every,
                    error_eval_points=error_eval_points,
                    additional_epochs_after_tolerance=5000
                )
    
                solver.train()
    
                device = solver.devices[0]
                final_losses = [solver.loss_history[device][i][-1] for i in range(len(test_M))]
                avg_final_loss = float(np.mean(final_losses))
    
                print(f"AVG final loss: {avg_final_loss:.3e}")
    
                if avg_final_loss < best_loss:
                    best_loss = avg_final_loss
                    best_config = (N, layers, activation)
                    best_run_dir = full_run_dir
    
            except Exception as e:
                print(f"FAILED config N={N}, layers={layers}, activation={activation} | error: {e}")
    
            finally:
                if "solver" in locals():
                    del solver
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Aggregate run summaries into CSV
    df = create_summary_dataframe(main_output_dir)
    df.to_csv(os.path.join(main_output_dir, "results_summary_validation_PINN_base_case.csv"), index=False)
    
    # Plot N-sensitivity
    plot_N_sensitivity(main_output_dir=main_output_dir, M_target=6.0,Gamma_target=0.4)

    print("\n================== BEST CONFIG ==================")
    print(f"Best config: {best_config}")
    print(f"Best avg final loss: {best_loss:.3e}")
    print(f"Best run dir: {best_run_dir}")
    print("=================================================\n")

    print("Saved CSV:", os.path.join(main_output_dir, "results_summary_validation_PINN_base_case.csv"))
    print("All outputs saved under:", main_output_dir)