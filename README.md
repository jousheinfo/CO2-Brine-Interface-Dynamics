# Stiff CO₂–Brine Sharp-Interface ODE (FDM vs PINNs + Transfer Learning)

This project solves a **highly nonlinear stiff ODE** governing the **dimensionless CO₂–brine interface** \(h_{aD}(\chi)\) dynamics, parameterized by the mobility ratio \(M\) and gravity number \(\Gamma\), under **immiscible flow** and **negligible capillarity** assumptions.

We compare:
- **Finite Difference Methods (FDM)**: forward / backward / central discretizations (tested in analytically benchmarked cases), and  
- **Physics-Informed Neural Networks (PINNs)**: validated against the analytical solution in the regime \(\Gamma < 0.5\), then optimized and extended via **transfer learning** to stiffer regimes.

> **Note:** This repository accompanies a submitted manuscript (under review).  
> Please cite the paper if you use any part of this code for research.

---

## Main Features

- **FDM benchmark (analytical regime)**  
  Forward, backward, and central finite-difference schemes tested for:
  - \([M,\Gamma]=[5.0,0.1]\)
  - \([M,\Gamma]=[5.0,0.3]\)  
  (Result: All FDM configurations tested failed to provide accurate/stable solutions in our experiments.)
  **Important clarification:** The FDM conclusions in this project apply only to the specific         numerical configurations tested here. More advanced numerical strategies, such as adaptive          meshing, implicit/semi-implicit schemes, preconditioning, or continuation methods, are not          implemented in this work.

- **PINN validation (analytical regime)**  
  Network: **[8,8,8]**, **tanh** hidden activation, **500** collocation points, **sigmoid output** to enforce \(0 \le h_{aD} \le 1\).  
  Validated against the FDM formulations and the analytical solution for:
  - \([M,\Gamma]=[5.0,0.1]\)
  - \([M,\Gamma]=[5.0,0.3]\)
  (Result: PINN framework successfully provided accurate/stable solutions in our experiments.)
 
- **PINN Base Case and Collocation-Point Sensitivity (extreme analytical regime)**  
  Base network: **[8,8,8]**, **tanh** hidden activation, **sigmoid output** to enforce \(0 \le h_{aD} \le 1\).
  Collocation-point sensitivity:
  - N = 10000
  - N = 50000
  - N = 100000
  Validated against the analytical solution for:
  - \([M,\Gamma]=[5.0,0.4]\)
  (Result: Indistinguishable predictions and training histories across these settings. Therefore, **N = 10,000** is adopted as the baseline collocation-point setting for       the subsequent optimization and transfer-learning studies.)

- **Targeted grid search (stiff benchmark case only)**  
  Grid search performed **only** for:
  - \([M,\Gamma]=[5.0,0.4]\)
  The grid search tests the following network widths:
  - [8, 8, 8]
  - [16, 16, 16]
  - [32, 32, 32]
  and the following activation functions:
  - tanh
  - SiLU
  - GELU
  Best-performing configuration: **[32,32,32] + tanh**.

- **Loss-weight sensitivity (optimized model)**  
  Compared:
  - Fixed weights: \(\lambda_{BC}=\lambda_{R}=\lambda_{IC}=1\)  
  - Adaptive gradient-based weighting  
  Result: **Fixed-weighting performed best** for \([5.0,0.4]\).

- **Transfer learning beyond analytical regime**  
  Transfer initialized from the final optimized model (**[32,32,32], tanh, fixed weights**).  
  Tested scenarios:
  1) \(M\) constant, increasing \(\Gamma\):  
     \([6.0,1.0]\), \([6.0,7.0]\), \([6.0,33.0]\)
  2) \(\Gamma\) constant, increasing \(M\):  
     \([6.0,1.0]\), \([10.0,1.0]\), \([15.0,1.0]\)
  3) Extreme stress test:  
     \([19.0,49.0]\)
  (Result: Transfer-learning PINN framework is more robust to increasing `M` than to increasing `Γ`. Increasing `Γ` produces a stronger deterioration in convergence and        constraint satisfaction, indicating that the gravity number is the dominant stiffness driver in this formulation. The extreme case `[19.0, 49.0]` fails, showing that        transfer learning alone is insufficient for strongly stiff regimes.)


---

## Dependencies

Tested with Python 3.x and:

- `numpy`
- `matplotlib`
- `scipy`
- `torch` (CUDA optional; only if you have Nvidia GPU)
- `pandas` (optional; only if you export tables)

Install (example):

```bash
pip install numpy matplotlib scipy torch pandas
