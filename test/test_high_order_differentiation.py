from mpi4py import MPI
import numpy as np
import basix
import ufl
from dolfinx import fem, mesh
from dolfinx_external_operator import (
    FEMExternalOperator,
    replace_external_operators,
)
from ufl import Measure, TestFunction, TrialFunction, derivative, split
from ufl.algorithms import expand_derivatives

def test_multiple_differentiation():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    element = basix.ufl.quadrature_element(
        domain.topology.cell_name(),
        degree=1,
        value_shape=(),
    )

    Q = fem.functionspace(domain, element)

    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("P", 1, (gdim,)))
    u = fem.Function(V, name="u")

    dx = Measure("dx", metadata={"quadrature_degree": 2})

    psi = FEMExternalOperator(u, function_space=Q, name="psi")
    E = psi * dx 
    F = ufl.derivative(E, u, ufl.TestFunction(V))
    F_expanded = expand_derivatives(expand_derivatives(F))
    F_replaced, F_external_operators = replace_external_operators(F_expanded)

    assert len(F_external_operators) == 1
    assert F_external_operators[0].derivatives == (1,)

    J = ufl.derivative(F_expanded, u, ufl.TrialFunction(V))
    J_expanded = expand_derivatives(expand_derivatives(J))
    J_replaced, J_external_operators = replace_external_operators(J_expanded)

    assert len(J_external_operators) == 1
    assert J_external_operators[0].derivatives == (2,)
