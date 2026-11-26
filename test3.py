import numpy as np
import matplotlib.pyplot as plt


from DFT_class import DFT1D
from DFT_class import estimate_convergence_order


# TEST 3: SCF
def test_dft_scf():
    print()
    print("=" * 70)
    print("TEST 3: SCF")
    print("=" * 70)

    L = 20.0

    # Referenz
    reference = DFT1D(L=L, N=800, order=8, soft_core=1.0, Z=1.0)

    #reference_result = (reference.solve_scf(max_iter=300, density_tol=1e-9, energy_tol=1e-11, mixing=0.3, eig_tol=1e-10))
    reference_result = (reference.solve_scf(max_iter=300, density_tol=1e-11, energy_tol=1e-13, mixing=0.3, eig_tol=1e-13))

    E_ref = (reference_result["energy"])

    print(f"\nreference energy = " f"{E_ref:.12e}")

    print(f"reference-SCF converged: " f"{reference_result['converged']}")

    print(f"reference-SCF iterations: " f"{reference_result['iterations']}")

    grid_sizes = [50, 100, 150, 200, 300, 400, 600]

    orders = [2, 4, 8]

    results = {}

    for order in orders:
        results[order] = {"dx": [], "energy": [], "error": [], "iterations": []}


    # SCF
    for N in grid_sizes:
        print()
        print(f"N = {N}")

        for order in orders:
            solver = DFT1D(L=L, N=N, order=order, soft_core=1.0, Z=1.0)

            #result = (solver.solve_scf(max_iter=300, density_tol=1e-8, energy_tol=1e-10, mixing=0.3, eig_tol=1e-10))
            result = (solver.solve_scf(max_iter=300, density_tol=1e-10, energy_tol=1e-12, mixing=0.3, eig_tol=1e-12))

            E = result["energy"]

            error = abs(E - E_ref)

            results[order]["dx"].append(solver.dx)

            results[order]["energy"].append(E)

            results[order]["error"].append(error)

            results[order]["iterations"].append(result["iterations"])

            print(f"O({order}): " f"E = {E:.10f}, " f"error = {error:.4e}, " f"SCF = {result['iterations']}, " f"converged = " f"{result['converged']}")


    # observed order of convergence
    print()
    print("observed order of convergence:")

    for order in orders:
        p = estimate_convergence_order(results[order]["dx"], results[order]["error"])

        print(f"O({order}): " f"p =~ {p:.3f}")

    # energy convergence
    plt.figure(figsize=(12, 6))

    plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13})

    markers = {2: "o", 4: "s", 8: "^"}

    for order in orders:
        #plt.loglog(results[order]["dx"], results[order]["error"], marker=markers[order], linestyle="-", label=f"O({order})")
        plt.loglog(results[order]["dx"], results[order]["error"], marker=markers[order], linestyle="-", label=fr"$\mathcal{{O}}({order})$")
   
    
    limit_val = 1e-12
    plt.axhline(y=limit_val, color='gray', linestyle='--', label='Error threshold')
    plt.ylim(bottom=1e-14)
    ymin, ymax = plt.ylim()
    plt.axhspan(ymin, limit_val, color='gray', alpha=0.2, label='Numerical noise')

    plt.xlabel(r"grid distance $\Delta x$")

    plt.ylabel(r"$|E-E_{\mathrm{ref}}|$")

    plt.title(r"convergence $T + V_{ext} + V_H + V_{XC}$")

    plt.grid(True, which="both", linestyle="--")

    plt.legend(loc='upper right')

    plt.gca().invert_xaxis()

    plt.tight_layout()

    #plt.show()
    plt.savefig("figures/scf.svg", dpi=300)
    plt.savefig("figures/scf.png", dpi=300)

    # density
    plt.figure(figsize=(9, 6))

    plt.plot(reference_result["x"], reference_result["density"], label=r"reference $\mathcal{O}(8)$, N=800")

    solver = DFT1D(L=L, N=600, order=8, soft_core=1.0, Z=1.0)
    result = (solver.solve_scf(max_iter=300, density_tol=1e-8, energy_tol=1e-10, mixing=0.3, eig_tol=1e-10))

    plt.plot(result["x"], result["density"], "--", label=r"$\mathcal{O}(8)$, N=600")

    solver_o2 = DFT1D(L=L, N=100, order=2, soft_core=1.0, Z=1.0)
    result_o2 = solver_o2.solve_scf(max_iter=300, density_tol=1e-8, energy_tol=1e-10, mixing=0.3, eig_tol=1e-10)
    plt.plot(result_o2["x"], result_o2["density"], ":", color="orange", label=r"$\mathcal{O}(2)$, N=100")

    plt.xlabel("x")

    plt.ylabel(r"$\rho(x)$")

    plt.title("electron density")

    plt.grid(True)

    plt.legend(loc='upper right')

    plt.tight_layout()

    #plt.show()
    plt.savefig("figures/density.svg", dpi=800)
    plt.savefig("figures/density.png", dpi=300)

    return results



def plot_test():
    L = 20.0
    #reference = DFT1D(L=L, N=800, order=8, soft_core=1.0, Z=1.0)
    #reference_result = (reference.solve_scf(max_iter=300, density_tol=1e-11, energy_tol=1e-13, mixing=0.3, eig_tol=1e-13))

    plt.figure(figsize=(9, 6))
    
    #plt.plot(reference_result["x"], reference_result["density"], label="reference O(8), N=800")
    
    solver = DFT1D(L=L, N=600, order=8, soft_core=1.0, Z=1.0)
    result = (solver.solve_scf(max_iter=300, density_tol=1e-8, energy_tol=1e-10, mixing=0.3, eig_tol=1e-10))
    plt.plot(result["x"], result["density"], "--", label="O(8), N=600", color="black")
    
    solver_o2 = DFT1D(L=L, N=100, order=2, soft_core=1.0, Z=1.0)
    result_o2 = solver_o2.solve_scf(max_iter=300, density_tol=1e-8, energy_tol=1e-10, mixing=0.3, eig_tol=1e-10)
    plt.plot(result_o2["x"], result_o2["density"], ":", color="orange", label="O(2), N=100")

    solver_o2_2 = DFT1D(L=L, N=10, order=2, soft_core=1.0, Z=1.0)
    result_o2_2 = solver_o2_2.solve_scf(max_iter=300, density_tol=1e-8, energy_tol=1e-10, mixing=0.3, eig_tol=1e-10)
    plt.plot(result_o2_2["x"], result_o2_2["density"], ":", color="blue", label="O(2), N=10")

    
    plt.xlabel("x")
    
    plt.ylabel(r"$\rho(x)$")
    
    plt.title("electron density")
    
    plt.grid(True)
    
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    plt.show()
    plt.savefig("figures/density.svg", dpi=800)
    plt.savefig("figures/density.png", dpi=300)


#plot_test()
#test_dft_scf()