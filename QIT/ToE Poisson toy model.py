"""
QIT / ToE Poisson Toy Model
---------------------------

Minimal reproducible sandbox:
1) Evolve a scalar information field s(x,y,t) by diffusion + damping
2) Build an *assumed* effective source term for a Poisson equation
3) Solve Poisson via FFT (periodic BC, zero-mean potential)
4) Compute g = -∇Φ and diagnostics; optional particle probes

DISCLAIMER:
- Toy mathematical model. Does NOT demonstrate new physics.
- "Repulsive/anti-gravity" refers only to local sign structure in the chosen
  effective Poisson source/divergence of g within this toy system.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt


Variant = Literal["original", "term_flipped"]


@dataclass
class Params:
    n: int = 256
    L: float = 10.0

    steps: int = 600
    D: float = 0.02
    Gamma: float = 0.01

    alpha: float = 1.0
    beta: float = 0.2
    lambda0: float = 0.0
    chi: float = 0.3
    variant: Variant = "term_flipped"
    mean_subtract_rhoinfo: bool = True

    rho_matter_amp: float = 0.0
    rho_matter_sigma: float = 0.7

    outdir: str = "outputs"
    seed: int = 0

    particles: int = 0
    particle_dt: float = 0.02
    particle_steps: int = 400

    time_proxy_eps: float = 1e-3
    time_proxy_clip: float = 5.0


def make_grid(n: int, L: float) -> Tuple[np.ndarray, np.ndarray, float]:
    dx = L / n
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    return X, Y, dx


def grad_periodic(f: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    fx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)
    fy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)
    return fx, fy


def laplacian_periodic(f: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(f, -1, axis=0)
        + np.roll(f, 1, axis=0)
        + np.roll(f, -1, axis=1)
        + np.roll(f, 1, axis=1)
        - 4 * f
    ) / (dx * dx)


def gaussian_blob(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(X * X + Y * Y) / (2 * sigma * sigma))


def fft_poisson_periodic(rhs: np.ndarray, L: float) -> np.ndarray:
    """
    Solve ∇² Φ = rhs on periodic 2D grid using FFT.
    Enforces zero-mean potential: Φ_hat(k=0)=0.
    """
    n = rhs.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KX, KY = np.meshgrid(k, k, indexing="xy")
    K2 = KX * KX + KY * KY

    rhs_hat = np.fft.fft2(rhs)
    phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)

    mask = K2 > 0
    phi_hat[mask] = -rhs_hat[mask] / K2[mask]
    phi_hat[~mask] = 0.0

    return np.real(np.fft.ifft2(phi_hat))


def evolve_s(s: np.ndarray, p: Params, dx: float) -> np.ndarray:
    # Explicit diffusion stability: dt <= dx^2/(4D)
    dt = min(0.2 * dx * dx / max(p.D, 1e-12), 0.2)
    for _ in range(p.steps):
        s = s + dt * (p.D * laplacian_periodic(s, dx) - p.Gamma * s)
    return s


def build_sources(s: np.ndarray, p: Params, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx, sy = grad_periodic(s, dx)
    grad2 = sx * sx + sy * sy

    rho_info = p.alpha * s + p.beta * grad2
    if p.mean_subtract_rhoinfo:
        rho_info = rho_info - np.mean(rho_info)

    lambda_info = p.lambda0 + p.chi * s

    # Optional toy matter term (centered Gaussian)
    n = s.shape[0]
    X, Y, _ = make_grid(n, p.L)
    rho_matter = p.rho_matter_amp * gaussian_blob(X, Y, p.rho_matter_sigma)
    rho_matter = rho_matter - np.mean(rho_matter)

    return rho_matter, rho_info, lambda_info


def rhs_from_variant(rho_m: np.ndarray, rho_i: np.ndarray, lam: np.ndarray, variant: Variant) -> np.ndarray:
    if variant == "term_flipped":
        return rho_m + rho_i + 0.5 * lam
    if variant == "original":
        return rho_m + rho_i - 0.5 * lam
    raise ValueError(f"Unknown variant: {variant}")


def diagnostics(phi: np.ndarray, dx: float) -> dict:
    phix, phiy = grad_periodic(phi, dx)
    gx, gy = -phix, -phiy
    gmag = np.sqrt(gx * gx + gy * gy)

    dgx_dx, _ = grad_periodic(gx, dx)
    _, dgy_dy = grad_periodic(gy, dx)
    divg = dgx_dx + dgy_dy

    return {
        "max_g": float(np.max(gmag)),
        "mean_g": float(np.mean(gmag)),
        "frac_divg_pos": float(np.mean(divg > 0.0)),
        "gmag_p50": float(np.percentile(gmag, 50)),
        "gmag_p90": float(np.percentile(gmag, 90)),
        "gmag_p99": float(np.percentile(gmag, 99)),
    }


def time_proxy(phi: np.ndarray, eps: float, clip: float) -> np.ndarray:
    """Visualization proxy derived from Φ. NOT GR time dilation."""
    denom = np.sqrt(np.maximum(1.0 + phi, eps))
    tau = 1.0 / denom
    return np.clip(tau, 0.0, clip)


def plot_fields(outpath: str, s0: np.ndarray, s: np.ndarray, rho_info: np.ndarray,
                phi: np.ndarray, dx: float, p: Params) -> None:
    phix, phiy = grad_periodic(phi, dx)
    gx, gy = -phix, -phiy
    gmag = np.sqrt(gx * gx + gy * gy)
    tp = time_proxy(phi, p.time_proxy_eps, p.time_proxy_clip)

    fig, axs = plt.subplots(2, 3, figsize=(12, 7))

    im = axs[0, 0].imshow(s0, origin="lower")
    axs[0, 0].set_title("Initial s (Gaussian)")
    plt.colorbar(im, ax=axs[0, 0], fraction=0.046)

    im = axs[0, 1].imshow(s, origin="lower")
    axs[0, 1].set_title("Final s")
    plt.colorbar(im, ax=axs[0, 1], fraction=0.046)

    im = axs[0, 2].imshow(rho_info, origin="lower")
    axs[0, 2].set_title("rho_info (effective)")
    plt.colorbar(im, ax=axs[0, 2], fraction=0.046)

    im = axs[1, 0].imshow(phi, origin="lower")
    axs[1, 0].set_title("Phi (potential)")
    plt.colorbar(im, ax=axs[1, 0], fraction=0.046)

    im = axs[1, 1].imshow(gmag, origin="lower")
    axs[1, 1].set_title("|g|")
    plt.colorbar(im, ax=axs[1, 1], fraction=0.046)

    im = axs[1, 2].imshow(tp, origin="lower")
    axs[1, 2].set_title("Time proxy (visual; NOT GR)")
    plt.colorbar(im, ax=axs[1, 2], fraction=0.046)

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"QIT/ToE Poisson Toy — variant={p.variant} mean_subtract={p.mean_subtract_rhoinfo}", y=0.98)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_g_quiver(outpath: str, phi: np.ndarray, dx: float) -> None:
    phix, phiy = grad_periodic(phi, dx)
    gx, gy = -phix, -phiy
    gmag = np.sqrt(gx * gx + gy * gy)

    n = phi.shape[0]
    step = max(n // 32, 1)
    yy, xx = np.mgrid[0:n:step, 0:n:step]

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(gmag, origin="lower")
    plt.quiver(xx, yy, gx[yy, xx], gy[yy, xx], color="white", alpha=0.7, scale=50)
    plt.title("Gravitational field direction (toy)\n(arrows show g; sign depends on assumed sources)")
    plt.colorbar(label="|g|")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close(fig)


def run_particles(phi: np.ndarray, p: Params, dx: float) -> dict:
    if p.particles <= 0:
        return {}

    phix, phiy = grad_periodic(phi, dx)
    gx, gy = -phix, -phiy
    n = phi.shape[0]

    rng = np.random.default_rng(p.seed + 123)
    pos = rng.normal(loc=n / 2, scale=n / 16, size=(p.particles, 2))
    vel = np.zeros_like(pos)

    ys0 = pos[:, 1].copy()
    for _ in range(p.particle_steps):
        iy = np.mod(np.round(pos[:, 1]).astype(int), n)
        ix = np.mod(np.round(pos[:, 0]).astype(int), n)
        ax = gx[iy, ix]
        ay = gy[iy, ix]
        vel[:, 0] += p.particle_dt * ax
        vel[:, 1] += p.particle_dt * ay
        pos += p.particle_dt * vel

    dy = pos[:, 1] - ys0
    return {"mean_dy": float(np.mean(dy)), "std_dy": float(np.std(dy))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--D", type=float, default=0.02)
    ap.add_argument("--Gamma", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--lambda0", type=float, default=0.0)
    ap.add_argument("--chi", type=float, default=0.3)
    ap.add_argument("--variant", choices=["original", "term_flipped"], default="term_flipped")
    ap.add_argument("--mean_subtract_rhoinfo", action="store_true")
    ap.add_argument("--no_mean_subtract_rhoinfo", action="store_true")
    ap.add_argument("--rho_matter_amp", type=float, default=0.0)
    ap.add_argument("--rho_matter_sigma", type=float, default=0.7)
    ap.add_argument("--particles", type=int, default=0)
    ap.add_argument("--particle_steps", type=int, default=400)
    ap.add_argument("--particle_dt", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mean_sub = (not args.no_mean_subtract_rhoinfo)
    if args.mean_subtract_rhoinfo:
        mean_sub = True

    p = Params(
        n=args.n, L=args.L, steps=args.steps, D=args.D, Gamma=args.Gamma,
        alpha=args.alpha, beta=args.beta, lambda0=args.lambda0, chi=args.chi,
        variant=args.variant, mean_subtract_rhoinfo=mean_sub,
        rho_matter_amp=args.rho_matter_amp, rho_matter_sigma=args.rho_matter_sigma,
        particles=args.particles, particle_steps=args.particle_steps, particle_dt=args.particle_dt,
        seed=args.seed
    )

    os.makedirs(p.outdir, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = os.path.join(p.outdir, run_id)
    os.makedirs(out, exist_ok=True)

    X, Y, dx = make_grid(p.n, p.L)
    s0 = gaussian_blob(X, Y, sigma=0.6)
    s = evolve_s(s0.copy(), p, dx)

    rho_m, rho_i, lam = build_sources(s, p, dx)
    rhs = rhs_from_variant(rho_m, rho_i, lam, p.variant)
    phi = fft_poisson_periodic(rhs, p.L)

    mets = diagnostics(phi, dx)
    pmets = run_particles(phi, p, dx)

    payload = {
        "params": asdict(p),
        "metrics": mets,
        "particle_metrics": pmets,
        "notes": {
            "disclaimer": "Toy model. 'Repulsive' refers to local divergence sign under assumed effective sources.",
            "poisson_rhs_mean": float(np.mean(rhs)),
            "phi_mean": float(np.mean(phi)),
        },
    }
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    plot_fields(os.path.join(out, "fields.png"), s0, s, rho_i, phi, dx, p)
    plot_g_quiver(os.path.join(out, "g_quiver.png"), phi, dx)

    print("=== DONE ===")
    print("Output directory:", out)
    print(f"max|g|={mets['max_g']:.6g}  frac(div g > 0)={mets['frac_divg_pos']:.3f}")
    if pmets:
        print(f"particle mean Δy={pmets['mean_dy']:.6g}  std Δy={pmets['std_dy']:.6g}")


if __name__ == "__main__":
    main()
