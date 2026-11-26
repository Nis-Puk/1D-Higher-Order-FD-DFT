import numpy as np
from DFT_class import DFT1D


from test_controll import test_kinetic_properties
from test1 import test_kinetic_operator
from test2 import test_external_hamiltonian
from test3 import test_dft_scf


print()
print("#"* 70)
print("# poster project NE451")
print("#"* 70)

# test test
test_kinetic_properties()

# test 1
kinetic_results = (test_kinetic_operator())

# test 2
external_results = (test_external_hamiltonian())

# test 3
dft_results = (test_dft_scf())

print()
print("#"* 70)
print("done")
print("#"* 70)