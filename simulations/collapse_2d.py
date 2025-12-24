"""
2D F-Fabric collapse simulation.
Demonstrates emergent horizon formation as χ-threshold contour.
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
N = 200          # grid size
T = 160          # time steps
CHI_H = 0.10     # horizon threshold

Amin = 0.05
A_thr = 1.0

g0 = 0.012
rg = 20

kOmega = 0.018
kA = 0.012
dOmega = 0.93

wL = 0.35
wC = 0.65
Dchi = 0.12

np.random.seed(1)

# =========================
# GRID
# =========================
y, x = np.mgrid[:N, :N]
cx = cy = N // 2
r = np.sqrt((x - cx)**2 + (y - cy)**2)
R = N // 3

# =========================
# INITIAL CONDITIONS
# =========================
A = np.zeros((N, N))
Omega = np.zeros_like(A)

mask = r < R
A[mask] = 0.85 + 0.05 * np.random.rand(mask.sum())
Omega[mask] = 1.0

# asymmetry
theta = np.arctan2(y - cy, x - cx)
A += 0.05 * np.cos(2 * theta) * np.exp(-r / R)

# =========================
# OPERATORS
# =========================
def laplacian(f):
    return (
        np.roll(f, 1, 0) + np.roll(f, -1, 0) +
        np.roll(f, 1, 1) + np.roll(f, -1, 1) -
        4 * f
    )

# =========================
# TIME EVOLUTION
# =========================
plt.figure(figsize=(6, 6))

for t in range(T):

    # 1. central amplitude growth
    A += g0 * np.exp(-r / rg)

    # 2. resonance loss
    Omega -= kOmega * np.clip(A - A_thr, 0, None)

    # 3. collective resonance
    Omega_c = (
        np.roll(Omega, 1, 0) + np.roll(Omega, -1, 0) +
        np.roll(Omega, 1, 1) + np.roll(Omega, -1, 1)
    ) / 4.0

    Omega_eff = wL * Omega + wC * Omega_c

    # 4. connectivity
    chi = Omega_eff / np.maximum(A, Amin)

    # 5. propagation of destruction
    chi = chi + Dchi * laplacian(chi)

    # 6. collapse
    collapse = chi < CHI_H
    Omega[collapse] *= dOmega
    A[collapse] += kA

    # =====================
    # VISUALIZATION
    # =====================
    if t % 5 == 0 or t == T - 1:
        plt.clf()

        field = np.clip((1.2 - chi) + 0.4 * A, 0, 1.5)
        plt.imshow(field, cmap="inferno", origin="lower")

        try:
            plt.contour(chi, levels=[CHI_H], colors="white", linewidths=1.2)
        except:
            pass

        plt.title(f"t = {t}")
        plt.axis("off")
        plt.pause(0.01)

plt.show()
