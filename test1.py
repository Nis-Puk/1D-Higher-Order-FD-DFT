import numpy as np
import matplotlib.pyplot as plt


from DFT_class import DFT1D
from DFT_class import estimate_convergence_order


# TEST 1: just kinetic operator
# L = 2.
#
# reason:
#
# L = 20 we have for O(8) reached eps -> bad
#
# with L = 2 we can observe asymptotic convercence O(2)/O(4)/O(8)
# numerisch sauber beobachten.
#
# Exakt: E_1 = pi**2 / (2 L**2)


def test_kinetic_operator():

    print()
    print("=" * 70)
    print("test 1: kinetic operator T")
    print("=" * 70)

    L = 2.0

    grid_sizes = [10, 14, 18, 24, 32, 44, 60, 80]

    orders = [2, 4, 8]

    results = {}

    for order in orders:
        results[order] = {"dx": [], "energy": [], "error": []}

    # analytic reference
    E_exact = (np.pi**2 / (2.0 * L**2))

    print(f"\nanalytic ground state: " f"E={E_exact:.12e}")

    for N in grid_sizes:
        print()
        print(f"N = {N}")

        for order in orders:
            solver = DFT1D(L=L, N=N, order=order)


            # eigenwerte kinetic operator
            # T im Sinusraum diagonal -> kleinste Eigenwert direkt der erste diskrete Modus
            E = float(solver.T_eigenvalues[0])

            error = abs(E - E_exact)

            results[order]["dx"].append(solver.dx)

            results[order]["energy"].append(E)

            results[order]["error"].append(error)

            print(f"O({order}): " f"E = {E:.12e}, " f"error = {error:.4e}")

    # observed order of convergence
    print()
    print("observed order of convergence:")

    for order in orders:
        p = estimate_convergence_order(results[order]["dx"], results[order]["error"])

        print(f"  O({order}): " f"p =~ {p:.3f}")

    # Plot
    plt.figure(figsize=(9, 6))

    markers = {2: "o", 4: "s", 8: "^"}

    for order in orders:

        #plt.loglog(results[order]["dx"], results[order]["error"], marker=markers[order], linestyle="-", label=f"O({order})")
        plt.loglog(results[order]["dx"], results[order]["error"], marker=markers[order], linestyle="-", label=fr"$\mathcal{{O}}({order})$")


    limit_val = 1e-13
    plt.axhline(y=limit_val, color='gray', linestyle='--', label='Error threshold')
    plt.ylim(bottom=1e-14)
    ymin, ymax = plt.ylim()
    plt.axhspan(ymin, limit_val, color='gray', alpha=0.2, label='Numerical noise')
    
    plt.xlabel(r"grid distance $\Delta x$")

    plt.ylabel(r"$|E_h-E_{\mathrm{exact}}|$")

    plt.title(r"convergence $T$")

    plt.grid(True, which="both", linestyle="--")

    plt.legend(loc='upper right')

    plt.gca().invert_xaxis()

    plt.tight_layout()

    #plt.show()
    plt.savefig("figures/kinetic_operator.svg", dpi=800)
    plt.savefig("figures/kinetic_operator.png", dpi=300)

    return results


#test_kinetic_operator()