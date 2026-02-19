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
  (All FDM configurations tested failed to provide accurate/stable solutions in our experiments.)

- **PINN base validation (analytical regime)**  
  Base network: **[16,16,16]**, **tanh** hidden activation, **sigmoid output** to enforce \(0 \le h_{aD} \le 1\).  
  Validated against the analytical solution for:
  - \([M,\Gamma]=[5.0,0.1]\)
  - \([M,\Gamma]=[5.0,0.4]\)

- **Targeted grid search (stiff benchmark case only)**  
  Grid search performed **only** for:
  - \([M,\Gamma]=[5.0,0.4]\)  
  Best-performing configuration: **[32,32,32] + tanh**.

- **Loss-weight sensitivity (optimized model)**  
  Compared:
  - Fixed weights: \(\lambda_{BC}=\lambda_{R}=\lambda_{IC}=1\)  
  - Adaptive gradient-based weighting  
  Result: **fixed weighting performed best** for \([5.0,0.4]\).

- **Transfer learning beyond analytical regime**  
  Transfer initialized from the final optimized model (**[32,32,32], tanh, fixed weights**).  
  Tested scenarios:
  1) \(M\) constant, increasing \(\Gamma\):  
     \([6.0,1.0]\), \([6.0,7.0]\), \([6.0,33.0]\)
  2) \(\Gamma\) constant, increasing \(M\):  
     \([6.0,1.0]\), \([10.0,1.0]\), \([15.0,1.0]\)
  3) Extreme stress test:  
     \([19.0,49.0]\)

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
