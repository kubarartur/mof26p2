import numpy as np
import matplotlib.pyplot as plt

meV_to_atm = 1 / 27211.4
"""conversion from milielectoronovolts to atomic units"""
mum_to_atm = 1 / 0.5292
"""conversion from micro meters to atomic units"""

m = 0.067
"""effective mass of electon in GaAs"""

a = 10*mum_to_atm
"""Parameter of the Poshl-Teller potential"""

nu = 3
"""Parameter of the Poshl-Teller potential"""

L = 200 * mum_to_atm
"""length of the potential well"""

N = 300
"""number of grid cells"""

dx = L / (N - 1)
"""grid step"""

alpha_crit = m*dx**2

xs = np.arange(N)*dx
V = -0.5/m/a**2*nu*(nu + 1)/(np.cosh((xs - 0.5*L)/a))**2


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
        psi -= project(psi, state)*state

    normPsi(psi)


def project(psi1: np.ndarray, psi2: np.ndarray):
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

        imagTstep(psi, alpha, orth=orth)
        energies.append(Eexpct(psi)/meV_to_atm)

        if info_every is not None:
            if step % info_every == 0:
                print(
                    f"step: {step}, energy_diff:"
                    + f"{abs(energies[-2] - energies[-1])/tol} tol"
                )

    print(f"converged after: {step}, iterations")
    return np.array(energies), psi, step


tol = 1e-9*meV_to_atm
info_every = int(1e4)
np.random.seed(42)
psi0 = (np.random.rand(N) - 0.5) * 2
psi0[0] = 0
psi0[-1] = 0

alpha = 0.95*alpha_crit

psi = psi0.copy()
normPsi(psi)
Es1, psi1, step1 = stepTillConvg(psi, alpha, tol)

psi = psi0.copy()
normPsi(psi)
Es2, psi2, step2 = stepTillConvg(psi, alpha, tol, orth=[psi1])

offset = -min(min(Es1), min(Es2)) + 1e-1

fig, ax = plt.subplots(1, 2)
ax = ax.ravel()
ax[0].plot(np.arange(Es1.shape[0]), Es1 + offset, label="stan podstawowy")
ax[0].plot(np.arange(Es2.shape[0]), Es2 + offset,
           label="pierwszy stan wzbudzony")

ax[1].plot(np.linspace(0, L/mum_to_atm, N), psi1, label="stan podstawowy")
ax[1].plot(np.linspace(0, L/mum_to_atm, N),
           psi2, label="pierwszy stan wzbudzony")

ax[0].set(xlabel="numer kroku", ylabel="wartość oczekiwana energii")
ax[1].set(xlabel="x [nm]", ylabel=r"$\psi$")

ax[0].semilogy()

ax[0].legend()
ax[1].legend()
plt.show()
