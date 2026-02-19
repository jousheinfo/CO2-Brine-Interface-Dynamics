# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 08:53:17 2026

@author: jpauya1
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib
import shutil


def is_okwen_valid(M, Gamma):
    """
    Check if M and Gamma values are within the valid range for Okwen correlation.
    Based on the correlation range: 5.0 < M < 20.0, 0.5 < Gamma < 50.0
    """
    return (5.0 < M < 20.0) and (0.5 < Gamma < 50.0)


# ----------------------------
# Analytical piecewise solution (FOR PLOTTING ONLY)
# ----------------------------
def h_analytical(x, M):
    """
    Piecewise analytical solution (Gamma does NOT appear in this closed form):
      h = 0                                for x <= 2/M
      h = (M - sqrt(2M/x)) / (M - 1)       for 2/M < x < 2M
      h = 1                                for x >= 2M
    """
    x = np.asarray(x, dtype=float)
    h = np.empty_like(x)

    x1 = 2.0 / M
    x2 = 2.0 * M

    mask1 = x <= x1
    mask2 = (x > x1) & (x < x2)
    mask3 = x >= x2

    h[mask1] = 0.0
    h[mask3] = 1.0
    h[mask2] = (M - np.sqrt(2.0 * M / x[mask2])) / (M - 1.0)
    return h


# ----------------------------
# Nonlinear BVP + Newton solver
# ----------------------------
class NonlinearInterfaceBVP:
    """
    Unknowns: u = [h0, h1, ..., h_{N-1}, xmax]  (size N+1)

    Equations:
      R0      = h0 - 0
      Ri      = ODE residual, i=1..N-2
      RN-1    = hN-1 - 1
      RN      = ∫_0^xmax (1-h) dx - 2

    scheme: "forward" | "backward" | "central"

    init_mode:
      "div" : h0 = (M - sqrt(2M/x)) / (M - 1)
      "mul" : h0 = (M - sqrt(2M/x)) * (M - 1)
    """

    def __init__(
        self,
        M,
        Gamma,
        N=500,
        scheme="central",
        tol=1e-5,
        max_iter=5000,
        alpha=0.2,
        init_mode="div",
        xmax0_factor=1.0
    ):
        self.M = float(M)
        self.Gamma = float(Gamma)
        self.N = int(N)
        self.scheme = scheme.lower().strip()
        if self.scheme not in ("forward", "backward", "central"):
            raise ValueError("scheme must be forward/backward/central")

        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)

        # ---- initial guess controls (stored for naming) ----
        self.init_mode = init_mode.lower().strip()
        if self.init_mode not in ("div", "mul"):
            raise ValueError("init_mode must be 'div' or 'mul'")
        self.xmax0_factor = float(xmax0_factor)

        # ---- initial guess xmax ----
        self.xmax0 = self.xmax0_factor * (2.0 * self.M)
        x0 = np.linspace(0.0, self.xmax0, self.N)

        eps = 1e-12
        x_safe = np.maximum(x0, eps)

        # ---- initial guess h0 (ONLY for the numerical solver) ----
        if self.init_mode == "div":
            h0 = (self.M - np.sqrt(2.0 * self.M / x_safe)) / (self.M - 1.0)
        else:  # "mul"
            h0 = (self.M - np.sqrt(2.0 * self.M / x_safe)) * (self.M - 1.0)

        # clip + enforce BCs
        h0 = np.clip(h0, 0.0, 1.0)
        h0[0] = 0.0
        h0[-1] = 1.0

        self.u = np.zeros(self.N + 1, dtype=float)
        self.u[:self.N] = h0
        self.u[self.N] = self.xmax0

        # bookkeeping
        self._done = False
        self._ok = False
        self._conv_it = None
        self.error_history = []

    def compute_chi_max(self):
        # Okwen correlation (your fit)
        return ((0.0324 * self.M - 0.0952) * self.Gamma
                + (0.1778 * self.M + 5.9682) * np.sqrt(self.Gamma)
                + 1.6962 * self.M - 3.0472)

    # ---- functions ----
    def _D(self, h):
        return h + self.M * (1.0 - h)

    def _F(self, h):
        return h / self._D(h)

    def _G(self, h):
        return (self.M * h * (1.0 - h)) / self._D(h)

    # ---- grid ----
    def grid(self, u):
        xmax = float(u[self.N])
        xmax = max(xmax, 1e-12)
        x = np.linspace(0.0, xmax, self.N)
        dx = x[1] - x[0]
        return x, dx, xmax

    # ---- ghost values (Dirichlet) ----
    @staticmethod
    def _hghost(i, h, N):
        if i < 0:
            return 0.0
        if i >= N:
            return 1.0
        return h[i]

    # ---- discrete derivatives ----
    def _dh(self, i, h, dx):
        N = self.N
        if self.scheme == "forward":
            return (self._hghost(i + 1, h, N) - self._hghost(i, h, N)) / dx
        if self.scheme == "backward":
            return (self._hghost(i, h, N) - self._hghost(i - 1, h, N)) / dx
        return (self._hghost(i + 1, h, N) - self._hghost(i - 1, h, N)) / (2.0 * dx)

    def _B(self, i, h, x, dx):
        hi = self._hghost(i, h, self.N)
        return self._F(hi) - self._G(hi) * (2.0 * self.Gamma * x[i]) * self._dh(i, h, dx)

    def _dB(self, i, h, x, dx):
        if self.scheme == "forward":
            return (self._B(i + 1, h, x, dx) - self._B(i, h, x, dx)) / dx
        if self.scheme == "backward":
            return (self._B(i, h, x, dx) - self._B(i - 1, h, x, dx)) / dx
        return (self._B(i + 1, h, x, dx) - self._B(i - 1, h, x, dx)) / (2.0 * dx)

    # ---- residual ----
    def residual(self, u):
        h = u[:self.N]
        x, dx, _ = self.grid(u)

        R = np.zeros(self.N + 1, dtype=float)

        R[0] = h[0] - 0.0
        R[self.N - 1] = h[self.N - 1] - 1.0

        for i in range(1, self.N - 1):
            FT = -x[i] * self._dh(i, h, dx)
            ST = 2.0 * self._dB(i, h, x, dx)
            R[i] = FT + ST

        R[self.N] = np.trapz(1.0 - h, x) - 2.0
        return R

    # ---- numerical Jacobian ----
    def jacobian(self, u, R0):
        n = self.N + 1
        J = np.zeros((n, n), dtype=float)
        for j in range(n):
            uj = u[j]
            step = 1e-7 * (1.0 + abs(uj))
            up = u.copy()
            up[j] = uj + step
            Rp = self.residual(up)
            J[:, j] = (Rp - R0) / step
        return J


# ----------------------------
# Plot helpers
# ----------------------------
def live_plot_solutions(solvers, schemes, colors, M, Gamma, it, chi_ok,
                        save_path=None, dpi=300):
    plt.clf()

    x_ana = np.linspace(1e-8, 2.0 * M, 500)
    plt.plot(x_ana, h_analytical(x_ana, M), "k-", linewidth=2.0, label="Analytical")

    for sch in schemes:
        s = solvers[sch]
        x, _, xmax = s.grid(s.u)
        h = s.u[:s.N]
        Rnow = s.residual(s.u)
        normR = np.linalg.norm(Rnow, ord=np.inf)
        status = "done" if s._done else "run "
        plt.plot(x, h, color=colors[sch],
                 label=f"{sch} [{status}] ||R||∞={normR:.1e}")

    # Plot Okwen point only if within valid range
    if is_okwen_valid(M, Gamma):
        plt.plot(chi_ok, 1.0, "ko", markersize=6, label=r"Okwen $\chi_{\max}$")

    plt.xlabel("x")
    plt.ylabel(r"$h_{aD}$")
    plt.title(f"M={M}, Γ={Gamma} | Iteration {it}")
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.pause(0.01)


def plot_final_solution_and_error_subplot(solvers, schemes, colors, M, Gamma, chi_ok,
                                         save_path=None, dpi=300):
    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    xmax_now = max(solvers[sch].grid(solvers[sch].u)[2] for sch in schemes)
    x_ana = np.linspace(1e-8, max(2.0 * M, xmax_now), 500)
    ax1.plot(x_ana, h_analytical(x_ana, M), "k-", linewidth=2.0, label="Analytical")

    for sch in schemes:
        s = solvers[sch]
        x, _, xmax = s.grid(s.u)
        h = s.u[:s.N]
        ax1.plot(x, h, color=colors[sch], label=f"{sch}")

    # Plot Okwen point only if within valid range
    if is_okwen_valid(M, Gamma):
        ax1.plot(chi_ok, 1.0, "ko", markersize=6, label=r"Okwen $\chi_{\max}$")
    
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$h_{aD}$")
    ax1.set_title(f"Final solution | M={M}, Γ={Gamma}")
    ax1.grid(True)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    for sch in schemes:
        s = solvers[sch]
        err = s.error_history
        if len(err) == 0:
            continue
        iters = np.arange(1, len(err) + 1)
        ax2.semilogy(iters, err, color=colors[sch], label=f"{sch}")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\|R\|_\infty$")
    ax2.set_title(f"Convergence history | M={M}, Γ={Gamma}")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


# ----------------------------
# Naming helpers
# ----------------------------
def _fmt_tag(val):
    s = f"{val:g}"
    s = s.replace(".", "p").replace("-", "m")
    return s


def make_pair_outdir(base_dir, M, Gamma):
    tag = f"M{_fmt_tag(M)}_G{_fmt_tag(Gamma)}"
    outdir = os.path.join(base_dir, tag)
    os.makedirs(outdir, exist_ok=True)
    return outdir, tag


def make_init_tag(solver: NonlinearInterfaceBVP) -> str:
    mode_tag = solver.init_mode
    fac_tag = _fmt_tag(solver.xmax0_factor)
    xmax0_tag = _fmt_tag(solver.xmax0)

    h0 = solver.u[:solver.N]
    h_digest = hashlib.md5(np.round(h0, 6).tobytes()).hexdigest()[:8]

    return f"init_{mode_tag}_fac{fac_tag}_xmax0_{xmax0_tag}_h{h_digest}"


# ----------------------------
# Driver
# ----------------------------
def run_pairs(
    pairs,
    N=500,
    tol=1e-5,
    max_iter=5000,
    plot_every=500,
    alpha=0.2,
    base_dir="runs_validation_FDM",
    init_list=(("div", 1.0)),
    clean_base_dir=True
):
    
    # ---- clean base_dir once per full run ----
    if clean_base_dir:
        if os.path.isdir(base_dir):
            shutil.rmtree(base_dir)   # removes EVERYTHING inside
        os.makedirs(base_dir, exist_ok=True)

    schemes = ["forward", "backward", "central"]
    colors = {"forward": "b", "backward": "r", "central": "g"}

    for (M, Gamma) in pairs:
        print(f"\n=== Pair: M={M}, Γ={Gamma} ===")

        # folder per (M, Gamma) pair
        outdir, tag = make_pair_outdir(base_dir, M, Gamma)

        # ---- summary text file (ONE per pair) ----
        summary_path = os.path.join(outdir, f"summary_{tag}.txt")

        with open(summary_path, "w", encoding="utf-8") as fsum:
            fsum.write(f"Summary for pair: M={M}, Gamma={Gamma}\n")
            fsum.write(f"N={N}, tol={tol}, max_iter={max_iter}, alpha={alpha}, plot_every={plot_every}\n")
            fsum.write("=" * 80 + "\n\n")

            for (init_mode, xmax0_factor) in init_list:
                header = f"INIT GUESS: init_mode={init_mode}, xmax0_factor={xmax0_factor}"
                print(f"--- {header} ---")
                fsum.write(header + "\n")
                fsum.write("-" * len(header) + "\n")

                solvers = {
                    sch: NonlinearInterfaceBVP(
                        M, Gamma, N=N, scheme=sch,
                        tol=tol, max_iter=max_iter, alpha=alpha,
                        init_mode=init_mode, xmax0_factor=xmax0_factor
                    )
                    for sch in schemes
                }

                chi_ok = solvers["central"].compute_chi_max()

                init_tag = make_init_tag(solvers["central"])
                prefix = f"{tag}_{init_tag}"

                plt.ion()
                plt.figure(figsize=(8, 5))
                last_saved_it = None

                for it in range(1, max_iter + 1):
                    for sch in schemes:
                        s = solvers[sch]
                        if s._done:
                            continue

                        u = s.u.copy()
                        u[0] = 0.0
                        u[s.N - 1] = 1.0
                        u[s.N] = max(u[s.N], 1e-6)

                        R = s.residual(u)
                        normR = np.linalg.norm(R, ord=np.inf)
                        s.error_history.append(normR)

                        if normR < s.tol:
                            s.u = u
                            s._done = True
                            s._ok = True
                            s._conv_it = it
                            continue

                        J = s.jacobian(u, R)
                        try:
                            du = np.linalg.solve(J, -R)
                        except np.linalg.LinAlgError:
                            s.u = u
                            s._done = True
                            s._ok = False
                            s._conv_it = it
                            continue

                        u_new = u + s.alpha * du

                        u_new[:s.N] = np.clip(u_new[:s.N], 0.0, 1.0)
                        u_new[0] = 0.0
                        u_new[s.N - 1] = 1.0
                        u_new[s.N] = max(u_new[s.N], 1e-6)

                        s.u = u_new

                    if it % plot_every == 0:
                        img_path = os.path.join(outdir, f"{prefix}_iter_{it:05d}.png")
                        live_plot_solutions(
                            solvers, schemes, colors, M, Gamma, it, chi_ok,
                            save_path=img_path
                        )
                        last_saved_it = it

                    if all(solvers[sch]._done for sch in schemes):
                        if last_saved_it != it:
                            img_path = os.path.join(outdir, f"{prefix}_iter_{it:05d}.png")
                            live_plot_solutions(
                                solvers, schemes, colors, M, Gamma, it, chi_ok,
                                save_path=img_path
                            )
                        break

                plt.ioff()

                # ---- summary (print + write to file) ----
                for sch in schemes:
                    s = solvers[sch]
                    x, _, xmax = s.grid(s.u)
                    h = s.u[:s.N]
                    integral_val = np.trapz(1.0 - h, x)
                    status = "OK" if s._ok else "NOT CONVERGED"
                    line = (f"{sch:8s}: {status:13s}  conv_it={s._conv_it}  "
                            f"xmax={xmax:.6g}  integral={integral_val:.6g}  alpha={s.alpha}")
                    print(line)
                    fsum.write(line + "\n")

                fsum.write("\n")

                final_path = os.path.join(outdir, f"{prefix}_final.png")
                plot_final_solution_and_error_subplot(
                    solvers, schemes, colors, M, Gamma, chi_ok,
                    save_path=final_path
                )

                plt.close("all")

        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    pairs = [(5.0, 0.1), (5.0, 0.3)]

    init_list = (
        ("div", 1.0),
        ("mul", 1.0),
    )

    run_pairs(
        pairs,
        N=500,
        tol=1e-5,
        max_iter=5000,
        plot_every=500,
        alpha=0.2,
        base_dir="runs_validation_FDM",
        init_list=init_list
    )
