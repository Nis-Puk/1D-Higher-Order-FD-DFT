import numpy as np
import matplotlib.pyplot as plt


from DFT_class import DFT1D
from DFT_class import estimate_convergence_order


# TEST 2: T + V_ext
def test_external_hamiltonian():

    print()
    print("=" * 70)
    print("TEST 2: T + V_ext")
    print("=" * 70)

    L = 20.0

    # Referenz O(8) with tight mesh
    reference = DFT1D(L=L, N=800, order=8, soft_core=1.0, Z=1.0)

    H_ref = (reference.build_external_hamiltonian())

    E_ref, psi_ref = (reference.ground_state(H_ref, tol=1e-11))

    print(f"\nreference energy="f"{E_ref:.12e}")

    grid_sizes = [50, 100, 150, 200, 300, 400, 600]

    orders = [2, 4, 8]

    results = {}

    for order in orders:
        results[order] = {"dx": [], "energy": [], "error": []}

    for N in grid_sizes:
        print()
        print(f"N = {N}")

        for order in orders:
            solver = DFT1D(L=L, N=N, order=order, soft_core=1.0, Z=1.0)

            H = (solver.build_external_hamiltonian())

            E, psi = (solver.ground_state(H, tol=1e-10))

            error = abs(E - E_ref)

            results[order]["dx"].append(solver.dx)

            results[order]["energy"].append(E)

            results[order]["error"].append(error)

            print(f"O({order}): " f"E = {E:.10f}, " f"error = {error:.4e}")

    # observed order of convergence
    print()
    print("observed order of convergence:")

    for order in orders:
        p = estimate_convergence_order(results[order]["dx"], results[order]["error"])

        print(f"O({order}): " f"p =~ {p:.3f}")


    # Plot
    plt.figure(figsize=(9, 6))

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


    plt.title(r"convergence $T + V_{ext}$")

    plt.grid(True, which="both", linestyle="--")

    plt.legend(loc='upper right')

    plt.gca().invert_xaxis()

    plt.tight_layout()

    #plt.show()
    plt.savefig("figures/T_Vext.svg", dpi=800)
    plt.savefig("figures/T_Vext.png", dpi=300)

    return results


#test_external_hamiltonian()