import numpy as np
import matplotlib.pyplot as plt
from DFT_class import DFT1D
from pyscf import gto, scf, fci, ao2mo


def run_pyscf_benchmark(dft_solver):
    n = dft_solver.N_int
    dx = dft_solver.dx
    
    # T + V_ext
    T_mat = np.column_stack([dft_solver.T @ vec for vec in np.eye(n)])
    h1e = T_mat + np.diag(dft_solver.V_ext)
    
    # Soft-Core-Coulomb
    dx_mat = dft_solver.x[:, None] - dft_solver.x[None, :]
    V_ee = 1.0 / np.sqrt(dx_mat**2 + dft_solver.soft_core**2)
    
    eri = np.zeros((n, n, n, n))
    idx = np.arange(n)
    eri[idx[:, None], idx[:, None], idx[None, :], idx[None, :]] = V_ee
    
    # PySCF Setup
    mol = gto.M()
    mol.nelectron = 2
    mol.spin = 0
    mol.incore_anyway = True
    
    # Restricted Hartree-Fock
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1e
    mf.get_ovlp = lambda *args: np.eye(n)
    mf._eri = ao2mo.restore(8, eri, n)
    e_rhf = mf.kernel()
    
    # RHF ecectron density
    dm_rhf = mf.make_rdm1()
    rho_rhf = np.diag(dm_rhf) / dx
    
    # exact solution
    cisolver = fci.FCI(mf)
    e_fci, fcivec = cisolver.kernel()
    
    # electron density in grid space
    dm_fci_mo = cisolver.make_rdm1(fcivec, n, 2)
    dm_fci_ao = mf.mo_coeff @ dm_fci_mo @ mf.mo_coeff.T
    rho_fci = np.diag(dm_fci_ao) / dx
    
    return {"e_rhf": e_rhf, "rho_rhf": rho_rhf, "e_fci": e_fci, "rho_fci": rho_fci}


def solve_own_exx(dft_solver):
    orig_vxc = dft_solver.xc_potential
    orig_exc = dft_solver.xc_energy_density
    
    dft_solver.xc_potential = lambda rho: -0.5 * dft_solver.hartree_potential(rho)
    dft_solver.xc_energy_density = lambda rho: -0.25 * rho * dft_solver.hartree_potential(rho)
    
    res = dft_solver.solve_scf()
    
    dft_solver.xc_potential = orig_vxc
    dft_solver.xc_energy_density = orig_exc
    return res



#solver_o2 = DFT1D(L=20.0, N=120, order=2, soft_core=1.0, Z=1.0)
#res_lda_o2 = solver_o2.solve_scf()
#res_exx_o2 = solve_own_exx(solver_o2)
#ref_o2 = run_pyscf_benchmark(solver_o2)

solver_o8 = DFT1D(L=20.0, N=120, order=8, soft_core=1.0, Z=1.0)
res_lda_o8 = solver_o8.solve_scf()
res_exx_o8 = solve_own_exx(solver_o8)
ref_o8 = run_pyscf_benchmark(solver_o8)


print("-" * 55)
print(f"1D Higher-Order FD DFT (EXX): {res_exx_o8['energy']:.10f} Ha")
print(f"PySCF RHF: {ref_o8['e_rhf']:.10f} Ha  (Diff: {abs(res_exx_o8['energy'] - ref_o8['e_rhf']):.2e})")
print(f"PySCF FCI (Exact): {ref_o8['e_fci']:.10f} Ha")
print(f"1D Higher-Order FD DFT (LDA): {res_lda_o8['energy']:.10f} Ha")
print("-" * 55)


plt.figure(figsize=(12, 6))

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13})


x = solver_o8.x

#lt.plot(x, res_exx_o2["density"], color="blue", linestyle="-", lw=1.5, label=r"Our Code (EXX) $\mathcal{O}(2)$")
#plt.plot(x, res_lda_o2["density"], color="blue", linestyle="--", lw=1.5, label=r"Our Code (1D-LDA) $\mathcal{O}(2)$")

plt.plot(x, ref_o8["rho_fci"], color="black", linestyle="-", lw=2.9, label=r"PySCF FCI")
plt.plot(x, ref_o8["rho_rhf"], color="black", linestyle="--", lw=2.9, alpha=0.8, label=r"PySCF RHF")
plt.plot(x, res_exx_o8["density"], color="#2ca02c", linestyle="-", lw=1.5, label=r"1D Higher-Order FD DFT (EXX) $\mathcal{O}(8)$")
plt.plot(x, res_lda_o8["density"], color="#2ca02c", linestyle="--", lw=1.5, label=r"1D Higher-Order FD DFT (1D-LDA) $\mathcal{O}(8)$")

#plt.title("Electron density $\\rho(x)$ in the 1D-Soft-Core-Potential")
plt.xlabel("$x$ ")
plt.ylabel("$\\rho(x)$")
plt.xlim(-10, 10)
plt.grid(True, alpha=0.3)
plt.legend()
    
plt.tight_layout()
plt.savefig("figures/pyscf_benchmark.svg", dpi=800)
#plt.show()
