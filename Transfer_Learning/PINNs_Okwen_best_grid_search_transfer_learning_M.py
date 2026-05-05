# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 10:21:47 2026

@author: jpauya1
<<<<<<< HEAD

This script implements transfer learning (warm-start) to train PINN models on new M
values using pre-trained models from previous M values as starting points.

Features:
1) Output folder created in same directory as script
2) Output root folder: runs_PINN_transfer_learning_M
3) Okwen correlation reference points for validation
4) Snapshot plots every 1000 epochs saved in snapshots/ subdirectory
5) Final plots saved outside snapshots/ at training completion:
   - LOSS_COMPONENTS_... (2x2 subplot showing individual loss components)
   - FINAL_4PANEL_... (solution + loss evolution)
6) Saves loss histories and solutions to JSON for analysis
=======
"""



# -*- coding: utf-8 -*-
"""
PINNs_Okwen_transfer_learning.py

TRANSFER LEARNING (Warm-start) for your PINN, MATCHING your GRID-SEARCH physics.

Improvements implemented:
1) Output folder is created in the SAME directory as this script.
2) Output root folder is named: runs_PINN_transfer_learning_gradient_based
3) Analytical solution removed from ALL plots (no analytical for Gamma >= 0.5).
4) In addition to snapshot plots every 1000 epochs saved inside snapshots/,
   save TWO final plots (only once, at the end) OUTSIDE snapshots/:
   - LOSS_COMPONENTS_... (2x2 loss components)
   - FINAL_4PANEL_...    (solution + physics losses + integral + total)

Note: Transfer learning varies M while keeping Gamma constant, using best model from grid search.
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
"""

import os
import re
import glob
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

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
<<<<<<< HEAD
        # Keep deterministic settings minimal for performance
        # torch.backends.cudnn.deterministic = True  # Commented out for speed
        # torch.backends.cudnn.benchmark = False  # Commented out for speed
=======
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75


# ============================================================
# Okwen correlation (for plotting the point only)
# ============================================================
def compute_chi_max_okwen(M, Gamma):
    return (0.0324 * M - 0.0952) * Gamma + (0.1778 * M + 5.9682) * (Gamma ** 0.5) + 1.6962 * M - 3.0472


def is_okwen_valid(M, Gamma):
    """
<<<<<<< HEAD
    Check if M and Gamma values are within valid range for Okwen correlation.
    Based on correlation range: 5.0 < M < 20.0, 0.5 < Gamma < 50.0
=======
    Check if M and Gamma values are within the valid range for Okwen correlation.
    Based on the correlation range: 0.5 <= M <= 10, 0.1 <= Gamma <= 0.5
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
    """
    return (5.0 < M < 20.0) and (0.5 < Gamma < 50.0)


# ============================================================
<<<<<<< HEAD
# PINN model definition
=======
# PINN model (same as your grid-search)
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
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
        modules.append(nn.Sigmoid())  # bound output to [0,1]
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
# Robust pretrained finder (prints folder contents)
# ============================================================
<<<<<<< HEAD
def find_best_pretrained_pair_folder(grid_search_root, run_dir_name):
    """Automatically find best available pretrained pair folder"""
    run_dir_path = os.path.join(grid_search_root, run_dir_name)
    
    if not os.path.exists(run_dir_path):
        raise FileNotFoundError(f"Run directory not found: {run_dir_path}")
    
    # Look for available pair folders
    pair_folders = []
    for item in os.listdir(run_dir_path):
        item_path = os.path.join(run_dir_path, item)
        if os.path.isdir(item_path) and item.startswith("pair_"):
            pair_folders.append(item)
    
    if not pair_folders:
        raise FileNotFoundError(f"No pair folders found in: {run_dir_path}")
    
    # Sort by pair number and take first available one
    pair_folders.sort()
    best_pair = pair_folders[0]
    
    print(f"🔍 Found available pairs: {pair_folders}")
    print(f"✅ Using best available pair: {best_pair}")
    
    return os.path.join(run_dir_path, best_pair)


=======
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
def find_pretrained_pth_from_pair_folder(pair_folder_path: str) -> str:
    p = Path(pair_folder_path)

    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Pair folder does not exist:\n{pair_folder_path}")

<<<<<<< HEAD
    print("\n🔍 DEBUG: pair folder path used by Python:")
    print("   ", str(p.resolve()))
    print("🔍 DEBUG: folder contents:")
=======
    print("\nDEBUG: pair folder path used by Python:")
    print("   ", str(p.resolve()))
    print("DEBUG: folder contents:")
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
    for item in sorted(p.iterdir()):
        print("   -", item.name)

    candidates = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in (".pth", ".pt")]

    if not candidates:
        raise FileNotFoundError(
            f"Pair folder found but no .pth/.pt inside:\n{pair_folder_path}\n"
            f"(If you see a .pth in File Explorer, then Python is not looking at the same folder.)"
        )

    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
<<<<<<< HEAD
    print("\n✅ Found checkpoint:")
=======
    print("\nFound checkpoint:")
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
    print("   ", str(candidates[0]))
    return str(candidates[0])


# ============================================================
# Transfer-learning solver (matches grid-search physics)
# ============================================================
class TransferLearningPINNSolver:
    def __init__(self,
                 M,
                 Gamma,
                 layers,
                 activation_function,
                 pretrained_path,
                 optimizer_name="Adam",
                 learning_rate=1e-5,
                 chi_min=0.0,
<<<<<<< HEAD
                 N=10000,
=======
                 N=100000,
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
                 epochs=50000,
                 tol=1e-5,
                 snapshot_every=1000,
                 additional_epochs_after_tolerance=5000,
                 output_root=None,
                 run_dir_name=None,
                 pair_index=1,
                 device=None,
                 init_seed=42,
                 use_dynamic_chi_max=False):

        self.M = float(M)
        self.Gamma = float(Gamma)

        self.layers = layers
        self.activation_function = activation_function
        self.optimizer_name = optimizer_name
        self.learning_rate = float(learning_rate)

        self.chi_min = float(chi_min)
        self.N = int(N)
        self.epochs = int(epochs)
        self.tol = float(tol)
        self.snapshot_every = int(snapshot_every)
        self.additional_epochs_after_tolerance = int(additional_epochs_after_tolerance)
        self.use_dynamic_chi_max = bool(use_dynamic_chi_max)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Fixed weighting (no adaptive weighting)
        self.adaptive_weighting = False
        self.loss_weights = {'ode': 1.0, 'boundary': 1.0, 'integral': 1.0}

        # Model
        self.model = PINN(self.layers, self.activation_function, init_seed=init_seed).to(self.device)

        # Load pretrained weights/biases
        self.pretrained_path = pretrained_path
        if pretrained_path is None or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained .pth not found:\n{pretrained_path}")

        ckpt = torch.load(pretrained_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        elif isinstance(ckpt, dict):
            self.model.load_state_dict(ckpt, strict=True)
        else:
            raise ValueError("Unsupported checkpoint format (expected dict).")

<<<<<<< HEAD
        print(f"✅ Loaded pretrained weights from:\n   {pretrained_path}")
=======
        print(f"Loaded pretrained weights from:\n   {pretrained_path}")
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75

        # Fresh optimizer
        self.optimizer = self._get_optimizer()

        # Domain init (like grid-search init)
        self.chi_max = 2.0 * self.M
        self.chi = torch.linspace(self.chi_min, self.chi_max, self.N).view(-1, 1).to(self.device)

        # Output dirs
        if run_dir_name is None:
            layers_str = str(self.layers).replace(" ", "")
            run_dir_name = f"run_layers_{layers_str}_act_{self.activation_function}"

        if output_root is None:
            raise ValueError("output_root must be provided (Path or str).")

        pair_dir_name = f"pair_{pair_index}_M{self.M}_Gamma{self.Gamma}"
        self.output_dir = os.path.join(str(output_root), run_dir_name, pair_dir_name)
        self.snapshot_dir = os.path.join(self.output_dir, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Histories
        self.loss_history = []
        self.ode_loss_history = []
        self.boundary_loss_history = []
        self.integral_loss_history = []

        self.tolerance_reached_epoch = None
        self.early_stop_epoch = None

<<<<<<< HEAD
        print("\n🚀 Transfer-learning solver initialized")
=======
        print("\nTransfer-learning solver initialized")
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
        print(f"   Target: M={self.M}, Γ={self.Gamma}")
        print(f"   Device: {self.device}")
        print(f"   χ_max(init)=2M={self.chi_max}")
        print(f"   Output: {self.output_dir}")

    def _get_optimizer(self):
        if self.optimizer_name == "Adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "SGD":
            return optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    # SAME physics loss as your grid search
    def loss_function(self, chi):
        chi = chi.requires_grad_(True).float()
        h_aD = self.model(chi)
        h_aD_chi = torch.autograd.grad(h_aD, chi, grad_outputs=torch.ones_like(h_aD), create_graph=True)[0]

        term1 = -chi * h_aD_chi
        F_h_aD = h_aD / (h_aD + self.M * (1 - h_aD))
        G_h_aD = self.M * h_aD * (1 - h_aD) / (h_aD + self.M * (1 - h_aD))
        P = F_h_aD - G_h_aD * 2 * self.Gamma * chi * h_aD_chi
        P_chi = torch.autograd.grad(P, chi, grad_outputs=torch.ones_like(P), create_graph=True)[0]
        term2 = 2 * P_chi
        ODE_loss = torch.mean((term1 + term2) ** 2)

        chi_min_tensor = torch.tensor([[self.chi_min]], dtype=torch.float, device=self.device)
        chi_max_tensor = torch.tensor([[self.chi_max]], dtype=torch.float, device=self.device)
        h_aD_min = self.model(chi_min_tensor)
        h_aD_max = self.model(chi_max_tensor)
        boundary_loss = (h_aD_min - 0) ** 2 + (h_aD_max - 1) ** 2

        integral_values = 1 - h_aD
        integral_result = torch.trapz(integral_values.squeeze(), chi.squeeze())
        integral_loss = torch.mean((integral_result - 2) ** 2)

        total_loss = ODE_loss + boundary_loss + integral_loss

        return total_loss, integral_result, ODE_loss, boundary_loss, integral_loss


    # ---------------------------
    # SNAPSHOT plot (saved in snapshots/)
    # 2x2: PINN solution + physics components + integral + total
    # NO analytical curve
    # ---------------------------
    def save_snapshot_4panel(self, epoch):
        # ---- color scheme (match your reference) ----
        c_pinn  = "blue"
        c_ode   = "red"
        c_bc    = "orange"
        c_int   = "green"
        c_total = "violet"   # magenta/pink (close to your figure)
    
        chi_np = self.chi.detach().cpu().numpy().squeeze()
        with torch.no_grad():
            h_pinn = self.model(self.chi).detach().cpu().numpy().squeeze()
    
        ode_loss = np.array(self.ode_loss_history, dtype=float)
        bc_loss  = np.array(self.boundary_loss_history, dtype=float)
        int_loss = np.array(self.integral_loss_history, dtype=float)
        tot_loss = np.array(self.loss_history, dtype=float)
        epochs_arr = np.arange(1, len(tot_loss) + 1)
    
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Epoch {epoch} | M={self.M}, Γ={self.Gamma} | Total Loss: {tot_loss[-1]:.2e}",
            fontsize=14
        )
    
        # --- Top-left: PINN solution + Okwen
        ax = axes[0, 0]
        ax.plot(chi_np, h_pinn, color=c_pinn, linewidth=2.5, label=f"PINN ($M={self.M}$, $\Gamma={self.Gamma}$)")
        
        # Only plot Okwen point if within valid range
        if is_okwen_valid(self.M, self.Gamma):
            okwen = compute_chi_max_okwen(self.M, self.Gamma)
            ax.plot(okwen, 1.0, "o", color="brown", markerfacecolor="green", markersize=8, label="Okwen")
        
        ax.set_xlabel(r"$\chi$")
        ax.set_ylabel(r"$h_{aD}(\chi)$")
        ax.set_title("PINN Solution")
        ax.grid(True, alpha=0.3)
        ax.legend()
    
        # --- Top-right: Physics loss components
        ax = axes[0, 1]
        ax.plot(epochs_arr, ode_loss, color=c_ode, linewidth=2.0, label="Residual")
        ax.plot(epochs_arr, bc_loss,  color=c_bc,  linewidth=2.0, label="Boundary Conditions")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Physics Loss")
        ax.set_title("Physics Loss Components")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    
        # --- Bottom-left: Integral loss
        ax = axes[1, 0]
        ax.plot(epochs_arr, int_loss, color=c_int, linewidth=2.0)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Integral Loss")
        ax.set_title(r"Integral Loss ($\int (1-h)\, d\chi = 2$)")
        ax.grid(True, which="both", alpha=0.3)
    
        # --- Bottom-right: Total loss
        ax = axes[1, 1]
        ax.plot(epochs_arr, tot_loss, color=c_total, linewidth=2.5)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title("Total Loss (Sum of All Losses)")
        ax.grid(True, which="both", alpha=0.3)
    
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
        out = os.path.join(self.snapshot_dir, f"snapshot_epoch_{epoch}.png")
        plt.savefig(out, dpi=250, bbox_inches="tight")
        plt.close(fig)


    # ---------------------------
    # FINAL plot 1: LOSS COMPONENTS (2x2) outside snapshots/
    # matches your first screenshot
    # ---------------------------
    # def save_final_loss_components_plot(self):
    #     ode_loss = np.array(self.ode_loss_history, dtype=float)
    #     bc_loss = np.array(self.boundary_loss_history, dtype=float)
    #     int_loss = np.array(self.integral_loss_history, dtype=float)
    #     tot_loss = np.array(self.loss_history, dtype=float)
    #     epochs_arr = np.arange(1, len(tot_loss) + 1)

    #     fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    #     fig.suptitle(f"Loss Components | M = {self.M}, Γ = {self.Gamma} ({self.device})", fontsize=16)

    #     # Residual
    #     ax = axes[0, 0]
    #     ax.plot(epochs_arr, ode_loss, linewidth=2.2)
    #     ax.set_yscale("log")
    #     ax.set_title("Residual Loss")
    #     ax.set_xlabel("Epoch")
    #     ax.set_ylabel("Residual Loss")
    #     ax.grid(True, which="both", alpha=0.3)

    #     # Boundary
    #     ax = axes[0, 1]
    #     ax.plot(epochs_arr, bc_loss, linewidth=2.2)
    #     ax.set_yscale("log")
    #     ax.set_title("Boundary Loss (h=0, h=1)")
    #     ax.set_xlabel("Epoch")
    #     ax.set_ylabel("Boundary Loss")
    #     ax.grid(True, which="both", alpha=0.3)

    #     # Integral
    #     ax = axes[1, 0]
    #     ax.plot(epochs_arr, int_loss, linewidth=2.2)
    #     ax.set_yscale("log")
    #     ax.set_title(r"Integral Loss ($\int (1-h)\, d\chi = 2$)")
    #     ax.set_xlabel("Epoch")
    #     ax.set_ylabel("Integral Loss")
    #     ax.grid(True, which="both", alpha=0.3)

    #     # Total
    #     ax = axes[1, 1]
    #     ax.plot(epochs_arr, tot_loss, linewidth=2.2)
    #     ax.set_yscale("log")
    #     ax.set_title("Total Loss")
    #     ax.set_xlabel("Epoch")
    #     ax.set_ylabel("Total Loss")
    #     ax.grid(True, which="both", alpha=0.3)

    #     plt.tight_layout(rect=[0, 0, 1, 0.95])

    #     out = os.path.join(self.output_dir, f"LOSS_COMPONENTS_pair_M{self.M}_Gamma{self.Gamma}_{self.device}.png")
    #     plt.savefig(out, dpi=300, bbox_inches="tight")
    #     plt.close(fig)

    # ---------------------------
    # FINAL plot 2: FINAL 4-PANEL outside snapshots/
    # matches your second screenshot (no analytical)
    # ---------------------------
    def save_final_4panel_plot(self, final_epoch):
        # Just reuse the same 4-panel layout as snapshot, but save outside snapshots with a distinct name.
        # We’ll regenerate it at final state:
        chi_np = self.chi.detach().cpu().numpy().squeeze()
        with torch.no_grad():
            h_pinn = self.model(self.chi).detach().cpu().numpy().squeeze()

        ode_loss = np.array(self.ode_loss_history, dtype=float)
        bc_loss = np.array(self.boundary_loss_history, dtype=float)
        int_loss = np.array(self.integral_loss_history, dtype=float)
        tot_loss = np.array(self.loss_history, dtype=float)
        epochs_arr = np.arange(1, len(tot_loss) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Epoch {final_epoch} | M={self.M}, Γ={self.Gamma} | Total Loss: {tot_loss[-1]:.2e}",
            fontsize=14
        )

        # PINN solution
        ax = axes[0, 0]
        ax.plot(chi_np, h_pinn, linewidth=2.2, label=f"PINN (M={self.M}, Γ={self.Gamma})")
        
        # Only plot Okwen point if within valid range
        if is_okwen_valid(self.M, self.Gamma):
            okwen = compute_chi_max_okwen(self.M, self.Gamma)
            ax.plot(okwen, 1.0, "o", markersize=8, label="Okwen")
        
        ax.set_xlabel(r"$\chi$")
        ax.set_ylabel(r"$h_{aD}(\chi)$")
        ax.set_title("PINN Solution")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Physics loss components
        ax = axes[0, 1]
        ax.plot(epochs_arr, ode_loss, linewidth=2.0, label="Residual (ODE)")
        ax.plot(epochs_arr, bc_loss, linewidth=2.0, label="Boundary Conditions")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Physics Loss")
        ax.set_title("Physics Loss Components")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        # Integral loss
        ax = axes[1, 0]
        ax.plot(epochs_arr, int_loss, linewidth=2.0)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Integral Loss")
        ax.set_title(r"Integral Loss ($\int(1-h)\, d\chi = 2$)")
        ax.grid(True, which="both", alpha=0.3)

        # Total loss
        ax = axes[1, 1]
        ax.plot(epochs_arr, tot_loss, linewidth=2.0)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title("Total Loss (Sum of All Losses)")
        ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out = os.path.join(self.output_dir, f"FINAL_4PANEL_pair_M{self.M}_Gamma{self.Gamma}_{self.device}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def save_final_files(self, final_epoch):
        # Save model
        model_path = os.path.join(self.output_dir, f"model_transfer_pair_M{self.M}_Gamma{self.Gamma}.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "M": self.M,
            "Gamma": self.Gamma,
            "layers": self.layers,
            "activation_function": self.activation_function,
            "optimizer_name": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "final_loss": float(self.loss_history[-1]) if self.loss_history else float("inf"),
            "epochs_trained": len(self.loss_history),
            "loss_weights_final": self.loss_weights,
            "pretrained_path": self.pretrained_path,
            "chi_min": self.chi_min,
            "chi_max": self.chi_max,
            "use_dynamic_chi_max": self.use_dynamic_chi_max
        }, model_path)

        # Save histories
        hist = {
            "total_loss": [float(x) for x in self.loss_history],
            "ode_loss": [float(x) for x in self.ode_loss_history],
            "boundary_loss": [float(x) for x in self.boundary_loss_history],
            "integral_loss": [float(x) for x in self.integral_loss_history],
            "epochs": list(range(1, len(self.loss_history) + 1)),
            "M": self.M,
            "Gamma": self.Gamma,
            "chi_min": self.chi_min,
            "chi_max": self.chi_max
        }
        with open(os.path.join(self.output_dir, "loss_histories.json"), "w") as f:
            json.dump(hist, f, indent=2)

        # Save solution data
        with torch.no_grad():
            chi_np = self.chi.detach().cpu().numpy().squeeze()
            h_pinn = self.model(self.chi).detach().cpu().numpy().squeeze()

        sol = {
            "chi": chi_np.tolist(),
            "h_pinn": h_pinn.tolist(),
            "M": self.M,
            "Gamma": self.Gamma
        }
        with open(os.path.join(self.output_dir, "solution_data.json"), "w") as f:
            json.dump(sol, f, indent=2)

        # Final plots (outside snapshots)
        self.save_final_4panel_plot(final_epoch)

<<<<<<< HEAD
        print(f"✅ Saved transfer-learning outputs to:\n   {self.output_dir}")

    def train(self):
        print("\n🎯 Training (transfer learning) started...")
=======
        print(f"Saved transfer-learning outputs to:\n   {self.output_dir}")

    def train(self):
        print("\nTraining (transfer learning) started...")
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
        t0 = time.time()

        final_epoch_ran = 0

        for epoch in range(1, self.epochs + 1):
            final_epoch_ran = epoch

            self.optimizer.zero_grad()
            loss, integral_result, ode_loss, boundary_loss, integral_loss = self.loss_function(self.chi)
            loss.backward()
            self.optimizer.step()

            self.loss_history.append(loss.item())
            self.ode_loss_history.append(ode_loss.item())
            self.boundary_loss_history.append(boundary_loss.item())
            self.integral_loss_history.append(integral_loss.item())

            # No adaptive weight updates needed

            # Early stop
            if self.tolerance_reached_epoch is None and loss.item() <= self.tol:
                self.tolerance_reached_epoch = epoch
                self.early_stop_epoch = epoch + self.additional_epochs_after_tolerance
                print(f"[EARLY_STOP] reached tol={self.tol:.0e} at epoch {epoch}; stop at {self.early_stop_epoch}")

            # Dynamic chi_max (optional, matches your grid-search behavior)
            if self.use_dynamic_chi_max:
                integral_error = (integral_result - 2.0).item()
                if integral_error > 0:
                    self.chi_max += 0.0001 * abs(integral_error)
                else:
                    self.chi_max -= 0.0001 * abs(integral_error)
                self.chi_max = max(self.chi_max, 1e-8)
                self.chi = torch.linspace(self.chi_min, self.chi_max, self.N).view(-1, 1).to(self.device)

            # Snapshots every 1000 epochs (saved inside snapshots/)
            if epoch == 1 or (self.snapshot_every and epoch % self.snapshot_every == 0):
                print(f"[{self.device}] epoch={epoch} | loss={loss.item():.2e}")
                self.save_snapshot_4panel(epoch)

            # Stop after additional epochs past tolerance
            if self.early_stop_epoch is not None and epoch >= self.early_stop_epoch:
                print(f"[EARLY_STOP] stopping at epoch {epoch}")
                break

        dt = time.time() - t0
<<<<<<< HEAD
        print(f"✅ Training completed in {dt:.2f} s")
=======
        print(f"Training completed in {dt:.2f} s")
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75

        # Save all final outputs + the two final plots outside snapshots/
        self.save_final_files(final_epoch_ran)


# ============================================================
# MAIN
# ============================================================
def main():
    # ------------------------------------------------------------------
    # Set global random seeds for reproducibility
    # ------------------------------------------------------------------
    set_random_seeds(42)
    
    # ------------------------------------------------------------------
    # 0) Output root: SAME folder where this script is located
    # ------------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    TRANSFER_OUT_ROOT = script_dir / "runs_PINN_best_grid_search_transfer_learning_M"

    # ------------------------------------------------------------------
    # NEW: delete ALL previous outputs and recreate root each run
    # ------------------------------------------------------------------
    if TRANSFER_OUT_ROOT.exists():
<<<<<<< HEAD
        print(f"🧹 Deleting existing output folder:\n   {TRANSFER_OUT_ROOT}")
        shutil.rmtree(TRANSFER_OUT_ROOT)
    TRANSFER_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"📁 Fresh output folder created:\n   {TRANSFER_OUT_ROOT}")
=======
        print(f"Deleting existing output folder:\n   {TRANSFER_OUT_ROOT}")
        shutil.rmtree(TRANSFER_OUT_ROOT)
    TRANSFER_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Fresh output folder created:\n   {TRANSFER_OUT_ROOT}")
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
    
    # ------------------------------------------------------------------
    # 1) WHERE YOUR GRID-SEARCH OUTPUTS LIVE (root)
    # ------------------------------------------------------------------
    GRID_SEARCH_ROOT = r"D:\Nonlinear_ODE\Validation\PINN\Grid_search\runs_validation_PINN_base_case_grid_search"

    BEST_LAYERS = [32, 32, 32]
    BEST_ACT = "tanh"
    RUN_DIR_NAME = f"run_layers_{str(BEST_LAYERS).replace(' ', '')}_act_{BEST_ACT}"

<<<<<<< HEAD
    # pretrained: automatically find the best available pair
    PRETRAINED_PAIR_FOLDER = find_best_pretrained_pair_folder(GRID_SEARCH_ROOT, RUN_DIR_NAME)

    pretrained_pth = find_pretrained_pth_from_pair_folder(PRETRAINED_PAIR_FOLDER)
    print("\n🎯 Using pretrained checkpoint:\n   ", pretrained_pth)
=======
    # pretrained specifically from pair 2: M=6.0, Gamma=0.4
    PRETRAINED_PAIR_FOLDER = os.path.join(
        GRID_SEARCH_ROOT,
        RUN_DIR_NAME,
        "pair_2_M6.0_Gamma0.4"
    )

    pretrained_pth = find_pretrained_pth_from_pair_folder(PRETRAINED_PAIR_FOLDER)
    print("\nUsing pretrained checkpoint:\n   ", pretrained_pth)
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75

    # ------------------------------------------------------------------
    # 2) TRANSFER TARGETS (EDIT THIS LIST)
    # ------------------------------------------------------------------
    targets = [
<<<<<<< HEAD
        {"M": 6.0, "Gamma": 1.0, "pair_index": 1},
        {"M": 10.0, "Gamma": 1.0, "pair_index": 2},
        {"M": 15.0, "Gamma": 1.0, "pair_index": 3},
=======
        {"M": 1.0, "Gamma": 0.4, "pair_index": 1},
        {"M": 6.0, "Gamma": 0.4, "pair_index": 2},
        {"M": 10.0, "Gamma": 0.4, "pair_index": 3},
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
    ]

    # ------------------------------------------------------------------
    # 3) TRAINING HYPERPARAMS (MATCH GRID SEARCH)
    # ------------------------------------------------------------------
<<<<<<< HEAD
    N = 10000
=======
    N = 100000
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
    epochs = 50000
    lr = 1e-5
    tol = 1e-5
    snapshot_every = 1000
    additional_epochs_after_tolerance = 5000

    USE_DYNAMIC_CHI_MAX = True  # matches your grid-search behavior

    # ------------------------------------------------------------------
    # 4) RUN TRANSFER LEARNING
    # ------------------------------------------------------------------
    for t in targets:
        print("\n" + "=" * 90)
        print(f"TRANSFER TRAINING: M={t['M']} | Γ={t['Gamma']}")
        print("=" * 90)

        solver = TransferLearningPINNSolver(
            M=t["M"],
            Gamma=t["Gamma"],
            layers=BEST_LAYERS,
            activation_function=BEST_ACT,
            pretrained_path=pretrained_pth,
            optimizer_name="Adam",
            learning_rate=lr,
            chi_min=0.0,
            N=N,
            epochs=epochs,
            tol=tol,
            snapshot_every=snapshot_every,
            additional_epochs_after_tolerance=additional_epochs_after_tolerance,
            output_root=str(TRANSFER_OUT_ROOT),   # SAME folder as script
            run_dir_name=RUN_DIR_NAME,
            pair_index=t["pair_index"],
            device=None,
            init_seed=42,
            use_dynamic_chi_max=USE_DYNAMIC_CHI_MAX
        )

        solver.train()

<<<<<<< HEAD
    print("\n🎉 All transfer-learning runs completed.")
=======
    print("\nAll transfer-learning runs completed.")
>>>>>>> a3aa7e2bb0a5f6dba1a84d97ac65c798bd6d7f75
    print("Outputs saved under:\n  ", str(TRANSFER_OUT_ROOT))


if __name__ == "__main__":
    main()
