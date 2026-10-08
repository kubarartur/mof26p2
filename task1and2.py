import numpy as np
import matplotlib.pyplot as plt

meV_to_atm = 1 / 27211.4
"""conversion from milielectoronovolts to atomic units"""
mum_to_atm = 1 / 0.5292
"""conversion from micro meters to atomic units"""

m = 0.067
"""effective mass of electon in GaAs"""

L = 200 * mum_to_atm
"""length of the potential well"""

N = 300
"""number of grid cells"""

dx = L / N
"""grid step"""

alpha_crit = m*dx**2

V = np.zeros((N,))


def applyHamiltonian(psi):
    psi_new = (
        -0.5/m/dx**2*(np.roll(psi, -1) - 2*psi +
                      np.roll(psi, 1)) + V*psi
    )
    psi_new[0] = 0
    psi_new[-1] = 0

    return psi_new


def calcNorm(psi: np.ndarray) -> float:
    return np.sqrt(np.sum(psi**2)*dx)


def normPsi(psi: np.ndarray) -> None:
    psi /= calcNorm(psi)


def Eexpct(psi: np.ndarray) -> float:
    return np.sum(psi*applyHamiltonian(psi)*dx)


def imagTstep(psi: np.ndarray, alpha: float, orth=[]) -> None:
    psi -= alpha*applyHamiltonian(psi)

    for state in orth:
        psi -= project(psi, state)*psi

    normPsi(psi)


def project(psi1, psi2):
    return (np.sum(psi1*psi2)*dx)


def stepTillConvg(psi, alpha, tol, orth=[], info_every=None):

    print(f"Step till convg called with alpha= {alpha}")

    avg_len = 10
    energies = []
    for _ in range(avg_len + 1):
        imagTstep(psi, alpha, orth=orth)
        energies.append(Eexpct(psi)/meV_to_atm)

    step = avg_len + 1

    while abs(sum(energies[-avg_len:])/avg_len - energies[-1]) > tol:
        step += 1

        imagTstep(psi, alpha)
        energies.append(Eexpct(psi)/meV_to_atm)

        if info_every is not None:
            if step % info_every == 0:
                print(
                    f"step: {step}, energy_diff:"
                    + f"{abs(energies[-2] - energies[-1])/tol} tol"
                )

    print(f"converged after: {step}, iterations")
    return np.array(energies), psi, step


tol = 1e-6*meV_to_atm
info_every = int(1e4)
np.random.seed(42)
psi0 = (np.random.rand(N) - 0.5) * 2
psi0[0] = 0
psi0[-1] = 0

# crit_percentages = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
#                             0.95, 0.999, 1, 1.001, 1.05, 1.8])
crit_percentages = np.array([0.8, 0.9, 0.95, 0.999, 1, 1.05, 1.8])
alphas = alpha_crit * crit_percentages

Es = []
psis = []
steps = []

for alpha in alphas:
    psi = psi0.copy()
    normPsi(psi)

    E, psi, step = stepTillConvg(psi, alpha, tol)

    Es.append(E.copy()/meV_to_atm)
    psis.append(psi.copy())
    steps.append(step)

fig, ax = plt.subplots(1, 2)
ax = ax.ravel()
for i, crit_percentage in enumerate(crit_percentages):
    ax[0].plot(
        np.arange(Es[i].shape[0]),
        Es[i],
        label=r"$\alpha$ =" +
        f" {crit_percentages[i]}" + r" $\alpha_{crit}$",
    )
    ax[1].plot(
        crit_percentage,
        steps[i],
        "ro"
    )

ax[0].set(xlabel="Step", ylabel="Wartość oczekiwana energii [meV]")
ax[1].set(xlabel=r"$\alpha$ [$\alpha_{crit}$]",
          ylabel="kroki potrzebne do zbieżności")
ax[0].semilogy()
ax[0].legend()
plt.show()

fig, ax = plt.subplots(1, 2)
ax = ax.ravel()

for i, _ in enumerate(crit_percentages):
    if crit_percentages[i] < 1:
        ax[0].plot(
            np.linspace(0, L / mum_to_atm, psis[i].shape[0]),
            psis[i],
            label=r"$\alpha$ =" +
            f" {crit_percentages[i]}" + r" $\alpha_{crit}$",
        )
    else:
        ax[1].plot(
            np.linspace(0, L / mum_to_atm, psis[i].shape[0]),
            psis[i],
            label=r"$\alpha$ =" +
            f" {crit_percentages[i]}" + r" $\alpha_{crit}$",
        )

ax[0].set(xlabel="x [nm]", ylabel=r"$\psi$")
ax[1].set(xlabel="x [nm]", ylabel=r"$\psi$")
ax[0].legend()
ax[1].legend()
plt.show()
