import numpy as np
import matplotlib.pyplot as plt


from DFT_class import DFT1D


# test:
# 1. T is positiv
# 2. T is hermitesch


def test_kinetic_properties():

    print()
    print("=" * 70)
    print("test operators")
    print("=" * 70)

    for order in [2, 4, 8]:
        solver = DFT1D(L=20.0, N=100, order=order)

        eigenvalues = (solver.T_eigenvalues)

        print(f"O({order}): " f"lambda_min = " f"{np.min(eigenvalues):.12e}, " f"lambda_max = " f"{np.max(eigenvalues):.12e}")

        if np.min(eigenvalues) <= 0.0:

            raise RuntimeError(f"O({order}) besitzt " f"negative eigenwerte")

    print("all eigenwerte (relevant ones) are positive")