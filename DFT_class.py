import numpy as np
import scipy.linalg
from scipy.fft import dst
from scipy.sparse.linalg import LinearOperator, eigsh


# Hamiltonoperator: H = T + V_ext + V_H + V_xc
# Kinetik: T = -1/2 d**2/dx**2
# BCs: psi(-L/2) = psi(+L/2) = 0
# Finite-Difference O(2), O(4), O(8).


class DFT1D:

    def __init__(self, L=20.0, N=200, order=2, soft_core=1.0, Z=1.0):

        if order not in (2, 4, 8):
            raise ValueError("order has to be 2, 4 or 8.")

        if N < 10:
            raise ValueError("N has to be at least 10.")

        self.L = float(L)
        self.N = int(N)
        self.order = int(order)

        self.soft_core = float(soft_core)

        self.Z = float(Z)

        # grid
        self.x_full = np.linspace(-L / 2.0, L / 2.0, N)

        self.dx = (self.x_full[1] - self.x_full[0])

        # inner points
        # Dirichlet BC: psi(left) = psi(right) = 0
        self.x = (self.x_full[1:-1])

        self.N_int = len(self.x)

        # number electrons
        self.N_electrons = 2.0

        # kinetic operator as LinearOperator
        self.T_eigenvalues = (self._kinetic_eigenvalues())

        self.T = (self._build_kinetic_operator())

        # external potential
        self.V_ext = (self._build_external_potential())


    # modified wavenumber ala with the FD approximations
    def _modified_wavenumber_squared(self, theta):
        """
        approximation of k**2.
        theta = k h
        for theta -> 0: k_h**2 = k**2 + O(h^p)
        """

        if self.order == 2:

            return (2.0 - 2.0 * np.cos(theta))

        elif self.order == 4:

            return (2.5 - (8.0 / 3.0) * np.cos(theta) + (1.0 / 6.0) * np.cos(2.0 * theta))

        elif self.order == 8:

            return (205.0 / 72.0 - (16.0 / 5.0) * np.cos(theta) + (2.0 / 5.0) * np.cos(2.0 * theta) - (16.0 / 315.0) * np.cos(3.0 * theta) + (1.0 / 280.0) * np.cos(4.0 * theta))

        raise RuntimeError("invalid order.")


    # eigenwerte kinetic operator
    def _kinetic_eigenvalues(self):

        """
        Dirichlet: psi_m(x) propto sin(m pi (x+L/2)/L)
        with
            m = 1,...,N-2
        and
            theta_m = m pi / (N-1).
        kinetic operator is T = 1/2 k_h**2.
        """

        n = self.N_int

        mode = np.arange(1, n + 1, dtype=float)

        theta = (np.pi * mode / (n + 1.0))

        k2 = (self._modified_wavenumber_squared(theta) / self.dx**2)

        return 0.5 * k2


    # kinetic operator
    def _build_kinetic_operator(self):

        """
        T is diagonal: T_hat_m = lambda_m T_hat_m
        scipy's orthonormal DST-I is itself-invers
        """

        n = self.N_int
        eigenvalues = (self.T_eigenvalues)

        def matvec(v):

            v = np.asarray(v, dtype=float)

            v_hat = dst(v, type=1, norm="ortho")

            result_hat = (eigenvalues * v_hat)

            result = dst(result_hat, type=1, norm="ortho")

            return result

        return LinearOperator(shape=(n, n), matvec=matvec, rmatvec=matvec, dtype=float)


    # kinetic energy vector
    def apply_kinetic(self, psi):

        return self.T @ psi


    # external potential
    def _build_external_potential(self):

        """
        Soft-Core-Coulomb:

            V_ext(x) = -Z / sqrt(x**2 + a**2)
        """

        return (-self.Z / np.sqrt(self.x**2 + self.soft_core**2))


    # hamiltonian as linear opartator
    def build_hamiltonian(self, V):

        """
        H = T + V

        with a diagonal V
        """

        V = np.asarray(V, dtype=float)

        n = self.N_int

        def matvec(v):

            return (self.T @ v + V * v)

        return LinearOperator(shape=(n, n), matvec=matvec, rmatvec=matvec, dtype=float)


    # external hamiltonian
    def build_external_hamiltonian(self):

        return self.build_hamiltonian(self.V_ext)


    # ground state hamiltonian
    def ground_state(self, H, tol=1e-11, maxiter=5000):

        """
        smallest eigenvalue / ground state of H
        """

        eigenvalues, eigenvectors = (eigsh(H, k=1, which="SA", tol=tol, maxiter=maxiter))

        E = float(eigenvalues[0])

        psi = np.asarray(eigenvectors[:, 0])

        # integral |psi|**2 dx = 1
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)

        psi /= norm

        # sign convention
        if psi[np.argmax(np.abs(psi))] < 0:

            psi *= -1.0

        return E, psi


    # hartree potential
    def hartree_potential(self, rho):

        """
        1D Coulomb-potential: V_H(x) = integral rho(x') / sqrt((x-x')**2 a**2) dx'
        """

        rho = np.asarray(rho, dtype=float)

        # Differenzmatrix
        dx_matrix = (self.x[:, None] - self.x[None, :])

        kernel = (1.0 / np.sqrt(dx_matrix**2 + self.soft_core**2))

        V_H = (kernel @ rho) * self.dx

        return V_H


    # XC density
    def xc_energy_density(self, rho):

        """
        local model: e_xc(rho) = -1/3 rho**(3/2)
        """

        rho_safe = np.maximum(rho, 0.0)

        return (-1.0 / 3.0 * rho_safe**1.5)


    # XC potetnial
    def xc_potential(self, rho):

        """
        derivative: V_xc = d e_xc / d rho = -1/2 sqrt(rho)
        """

        rho_safe = np.maximum(rho, 0.0)

        return (-0.5 * np.sqrt(rho_safe))


    # start desity
    def initial_density(self):

        """
        starting density
        """

        rho = (np.exp(-self.x**2) / np.sqrt(np.pi))

        # zwei Elektronen normieren
        normalization = (np.sum(rho) * self.dx)

        rho *= (self.N_electrons / normalization)

        return rho


    # SCF
    def solve_scf(self, max_iter=200, density_tol=1e-12, energy_tol=1e-13, mixing=0.3, eig_tol=1e-12):

        rho = (self.initial_density())

        E_old = None

        converged = False

        final_eigenvalue = None
        final_psi = None

        for iteration in range(1, max_iter + 1):
            # Hartree
            V_H = (self.hartree_potential(rho))

            # XC
            V_xc = (self.xc_potential(rho))

            # effective potential
            V_eff = (self.V_ext + V_H + V_xc)

            # Hamiltonoperator
            H = (self.build_hamiltonian(V_eff))

            # ground state
            epsilon, psi = (self.ground_state(H, tol=eig_tol))

            final_eigenvalue = epsilon
            final_psi = psi

            # new density
            rho_new = (self.N_electrons * np.abs(psi)**2)

            # Mixing
            rho_mixed = (mixing * rho_new + (1.0 - mixing) * rho)

            # new potentials
            V_H_new = (self.hartree_potential(rho_mixed))

            V_xc_new = (self.xc_potential(rho_mixed))

            # Hartree energy
            E_H = (0.5 * np.sum(rho_mixed * V_H_new) * self.dx)

            # XC energie
            E_xc = (np.sum(self.xc_energy_density(rho_mixed)) * self.dx)

            # Summe KS Eigenwerte
            # two electrons in the same orbital
            E_KS = (self.N_electrons * epsilon)

            # XC double counting
            E_xc_potential = (np.sum(rho_mixed * V_xc_new) * self.dx)

            # E = sum epsilon_i - E_H + E_xc - integral rho V_xc
            E_total = (E_KS - E_H + E_xc - E_xc_potential)

            # error density
            density_error = np.max(np.abs(rho_mixed - rho))

            # error energy
            if E_old is None:
                energy_error = np.inf

            else:
                energy_error = abs(E_total - E_old)

            # Update
            rho = rho_mixed

            # convergence ?
            if (density_error < density_tol and energy_error < energy_tol):
                converged = True
                break

            E_old = E_total

        return {"energy": float(E_total), "density": rho, "x": self.x.copy(), "eigenvalue": float(final_eigenvalue), "psi": final_psi, "iterations": iteration, "converged": converged}


# order convergence
def estimate_convergence_order(dx, errors):

    dx = np.asarray(dx, dtype=float)

    errors = np.asarray(errors, dtype=float)

    mask = (np.isfinite(dx) & np.isfinite(errors) & (dx > 0) & (errors > 0))

    if np.count_nonzero(mask) < 2:
        return np.nan

    x = np.log(dx[mask])

    y = np.log(errors[mask])

    p = np.polyfit(x, y, 1)[0]

    return float(p)
