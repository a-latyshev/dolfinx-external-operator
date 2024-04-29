from mpi4py import MPI
from petsc4py import PETSc

import matplotlib.pyplot as plt
import numba
import numpy as np
import sys
sys.path.append("../docs/demos/")
import solvers
from solvers import LinearProblem
from utilities import build_cylinder_quarter, find_cell_by_point

import basix
import ufl
from dolfinx import common, fem
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

R_e, R_i = 1.3, 1.0  # external/internal radius

E, nu = 70e3, 0.3  # elastic parameters
E_tangent = E / 100.0  # tangent modulus
H = E * E_tangent / (E - E_tangent)  # hardening modulus
sigma_0 = 250.0  # yield strength

lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
mu = E / 2.0 / (1.0 + nu)
# stiffness matrix
C_elas = np.array(
    [
        [lmbda + 2.0 * mu, lmbda, lmbda, 0.0],
        [lmbda, lmbda + 2.0 * mu, lmbda, 0.0],
        [lmbda, lmbda, lmbda + 2.0 * mu, 0.0],
        [0.0, 0.0, 0.0, 2.0 * mu],
    ],
    dtype=PETSc.ScalarType,
)

deviatoric = np.eye(4, dtype=PETSc.ScalarType)
deviatoric[:3, :3] -= np.full((3, 3), 1.0 / 3.0, dtype=PETSc.ScalarType)

mesh, facet_tags, facet_tags_labels = build_cylinder_quarter()

k_u = 2
V = fem.functionspace(mesh, ("Lagrange", k_u, (2,)))
# Boundary conditions
bottom_facets = facet_tags.find(facet_tags_labels["Lx"])
left_facets = facet_tags.find(facet_tags_labels["Ly"])

bottom_dofs_y = fem.locate_dofs_topological(V.sub(1), mesh.topology.dim - 1, bottom_facets)
left_dofs_x = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, left_facets)

sym_bottom = fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), bottom_dofs_y, V.sub(1))
sym_left = fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), left_dofs_x, V.sub(0))

bcs = [sym_bottom, sym_left]


def epsilon(v):
    grad_v = ufl.grad(v)
    return ufl.as_vector([grad_v[0, 0], grad_v[1, 1], 0, np.sqrt(2.0) * 0.5 * (grad_v[0, 1] + grad_v[1, 0])])

k_stress = 2 * (k_u - 1)
ds = ufl.Measure(
    "ds",
    domain=mesh,
    subdomain_data=facet_tags,
    metadata={"quadrature_degree": k_stress, "quadrature_scheme": "default"},
)

dx = ufl.Measure(
    "dx",
    domain=mesh,
    metadata={"quadrature_degree": k_stress, "quadrature_scheme": "default"},
)

def via_numba(verbose=False):
    Du = fem.Function(V, name="displacement_increment")
    S_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=k_stress, value_shape=(4,))
    S = fem.functionspace(mesh, S_element)
    sigma = FEMExternalOperator(epsilon(Du), function_space=S)

    n = ufl.FacetNormal(mesh)
    loading = fem.Constant(mesh, PETSc.ScalarType(0.0))

    v = ufl.TestFunction(V)
    F = ufl.inner(sigma, epsilon(v)) * dx - loading * ufl.inner(v, n) * ds(facet_tags_labels["inner"])

    # Internal state
    P_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=k_stress, value_shape=())
    P = fem.functionspace(mesh, P_element)

    p = fem.Function(P, name="cumulative_plastic_strain")
    dp = fem.Function(P, name="incremental_plastic_strain")
    sigma_n = fem.Function(S, name="stress_n")

    num_quadrature_points = P_element.dim

    @numba.njit
    def return_mapping(deps_, sigma_n_, p_):
        """Performs the return-mapping procedure."""
        num_cells = deps_.shape[0]

        C_tang_ = np.empty((num_cells, num_quadrature_points, 4, 4), dtype=PETSc.ScalarType)
        sigma_ = np.empty_like(sigma_n_)
        dp_ = np.empty_like(p_)

        def _kernel(deps_local, sigma_n_local, p_local):
            """Performs the return-mapping procedure locally."""
            sigma_elastic = sigma_n_local + C_elas @ deps_local
            s = deviatoric @ sigma_elastic
            sigma_eq = np.sqrt(3.0 / 2.0 * np.dot(s, s))

            f_elastic = sigma_eq - sigma_0 - H * p_local
            f_elastic_plus = (f_elastic + np.sqrt(f_elastic**2)) / 2.0

            dp = f_elastic_plus / (3 * mu + H)

            n_elas = s / sigma_eq * f_elastic_plus / f_elastic
            beta = 3 * mu * dp / sigma_eq

            sigma = sigma_elastic - beta * s

            n_elas_matrix = np.outer(n_elas, n_elas)
            C_tang = C_elas - 3 * mu * (3 * mu / (3 * mu + H) - beta) * n_elas_matrix - 2 * mu * beta * deviatoric

            return C_tang, sigma, dp

        for i in range(0, num_cells):
            for j in range(0, num_quadrature_points):
                C_tang_[i, j], sigma_[i, j], dp_[i, j] = _kernel(deps_[i, j], sigma_n_[i, j], p_[i, j])

        return C_tang_, sigma_, dp_

    def C_tang_impl(deps):
        num_cells = deps.shape[0]
        num_quadrature_points = int(deps.shape[1] / 4)

        deps_ = deps.reshape((num_cells, num_quadrature_points, 4))
        sigma_n_ = sigma_n.x.array.reshape((num_cells, num_quadrature_points, 4))
        p_ = p.x.array.reshape((num_cells, num_quadrature_points))

        C_tang_, sigma_, dp_ = return_mapping(deps_, sigma_n_, p_)

        return C_tang_.reshape(-1), sigma_.reshape(-1), dp_.reshape(-1)


    def sigma_external(derivatives):
        if derivatives == (1,):
            return C_tang_impl
        else:
            return NotImplementedError


    sigma.external_function = sigma_external

    u_hat = ufl.TrialFunction(V)
    J = ufl.derivative(F, Du, u_hat)
    J_expanded = ufl.algorithms.expand_derivatives(J)

    F_replaced, F_external_operators = replace_external_operators(F)
    J_replaced, J_external_operators = replace_external_operators(J_expanded)

    eps = np.finfo(PETSc.ScalarType).eps
    Du.x.array[:] = eps

    timer1 = common.Timer("1st numba pass")
    timer1.start()
    evaluated_operands = evaluate_operands(F_external_operators)
    ((_, sigma_new, dp_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
    timer1.stop()

    timer2 = common.Timer("2nd numba pass")
    timer2.start()
    evaluated_operands = evaluate_operands(F_external_operators)
    ((_, sigma_new, dp_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
    timer2.stop()

    u = fem.Function(V, name="displacement")
    du = fem.Function(V, name="Newton_correction")
    external_operator_problem = LinearProblem(J_replaced, F_replaced, Du, bcs=bcs)

    x_point = np.array([[R_i, 0, 0]])
    cells, points_on_process = find_cell_by_point(mesh, x_point)

    q_lim = 2.0 / np.sqrt(3.0) * np.log(R_e / R_i) * sigma_0
    num_increments = 20
    max_iterations, relative_tolerance = 200, 1e-8
    load_steps = (np.linspace(0, 1.1, num_increments, endpoint=True) ** 0.5)[1:]
    loadings = q_lim * load_steps

    for i, loading_v in enumerate(loadings):
        loading.value = loading_v
        external_operator_problem.assemble_vector()

        residual_0 = residual = external_operator_problem.b.norm()
        Du.x.array[:] = 0.0

        if MPI.COMM_WORLD.rank == 0:
            print(f"\nresidual , {residual} \n increment: {i+1!s}, load = {loading.value}")

        for iteration in range(0, max_iterations):
            if residual / residual_0 < relative_tolerance:
                break
            external_operator_problem.assemble_matrix()
            external_operator_problem.solve(du)
            du.x.scatter_forward()

            Du.vector.axpy(-1.0, du.vector)
            Du.x.scatter_forward()

            evaluated_operands = evaluate_operands(F_external_operators)

            # Implementation of an external operator may return several outputs and
            # not only its evaluation. For example, `C_tang_impl` returns a tuple of
            # Numpy-arrays with values of `C_tang`, `sigma` and `dp`.
            ((_, sigma_new, dp_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)

            # In order to update the values of the external operator we may directly
            # access them and avoid the call of
            # `evaluate_external_operators(F_external_operators, evaluated_operands).`
            sigma.ref_coefficient.x.array[:] = sigma_new
            dp.x.array[:] = dp_new

            external_operator_problem.assemble_vector()
            residual = external_operator_problem.b.norm()

            if MPI.COMM_WORLD.rank == 0 and verbose:
                print(f"    it# {iteration} residual: {residual}")

        u.vector.axpy(1.0, Du.vector)
        u.x.scatter_forward()

        # Taking into account the history of loading
        p.vector.axpy(1.0, dp.vector)
        # skip scatter forward, p is not ghosted.
        # TODO: Why? What is the difference with lines above?
        sigma_n.x.array[:] = sigma.ref_coefficient.x.array
        # skip scatter forward, sigma is not ghosted.

    # TODO: Is there a more elegant way to extract the data?
    common.list_timings(MPI.COMM_WORLD, [common.TimingType.wall])

def via_interpolation_based(verbose=False):
    Du = fem.Function(V, name="displacement_increment")
    S_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=k_stress, value_shape=(4,))
    S = fem.functionspace(mesh, S_element)
    sigma = FEMExternalOperator(epsilon(Du), function_space=S)

    n = ufl.FacetNormal(mesh)
    loading = fem.Constant(mesh, PETSc.ScalarType(0.0))

    v = ufl.TestFunction(V)
    F = ufl.inner(sigma, epsilon(v)) * dx - loading * ufl.inner(v, n) * ds(facet_tags_labels["inner"])

    # Internal state
    P_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=k_stress, value_shape=())
    P = fem.functionspace(mesh, P_element)

    p = fem.Function(P, name="cumulative_plastic_strain")
    dp = fem.Function(P, name="incremental_plastic_strain")
    sigma_n = fem.Function(S, name="stress_n")

    num_quadrature_points = P_element.dim

    @numba.njit
    def return_mapping(deps_, sigma_n_, p_):
        """Performs the return-mapping procedure."""
        num_cells = deps_.shape[0]

        C_tang_ = np.empty((num_cells, num_quadrature_points, 4, 4), dtype=PETSc.ScalarType)
        sigma_ = np.empty_like(sigma_n_)
        dp_ = np.empty_like(p_)

        def _kernel(deps_local, sigma_n_local, p_local):
            """Performs the return-mapping procedure locally."""
            sigma_elastic = sigma_n_local + C_elas @ deps_local
            s = deviatoric @ sigma_elastic
            sigma_eq = np.sqrt(3.0 / 2.0 * np.dot(s, s))

            f_elastic = sigma_eq - sigma_0 - H * p_local
            f_elastic_plus = (f_elastic + np.sqrt(f_elastic**2)) / 2.0

            dp = f_elastic_plus / (3 * mu + H)

            n_elas = s / sigma_eq * f_elastic_plus / f_elastic
            beta = 3 * mu * dp / sigma_eq

            sigma = sigma_elastic - beta * s

            n_elas_matrix = np.outer(n_elas, n_elas)
            C_tang = C_elas - 3 * mu * (3 * mu / (3 * mu + H) - beta) * n_elas_matrix - 2 * mu * beta * deviatoric

            return C_tang, sigma, dp

        for i in range(0, num_cells):
            for j in range(0, num_quadrature_points):
                C_tang_[i, j], sigma_[i, j], dp_[i, j] = _kernel(deps_[i, j], sigma_n_[i, j], p_[i, j])

        return C_tang_, sigma_, dp_

    def C_tang_impl(deps):
        num_cells = deps.shape[0]
        num_quadrature_points = int(deps.shape[1] / 4)

        deps_ = deps.reshape((num_cells, num_quadrature_points, 4))
        sigma_n_ = sigma_n.x.array.reshape((num_cells, num_quadrature_points, 4))
        p_ = p.x.array.reshape((num_cells, num_quadrature_points))

        C_tang_, sigma_, dp_ = return_mapping(deps_, sigma_n_, p_)

        return C_tang_.reshape(-1), sigma_.reshape(-1), dp_.reshape(-1)


    def sigma_external(derivatives):
        if derivatives == (1,):
            return C_tang_impl
        else:
            return NotImplementedError


    sigma.external_function = sigma_external

    u_hat = ufl.TrialFunction(V)
    J = ufl.derivative(F, Du, u_hat)
    J_expanded = ufl.algorithms.expand_derivatives(J)

    F_replaced, F_external_operators = replace_external_operators(F)
    J_replaced, J_external_operators = replace_external_operators(J_expanded)

    eps = np.finfo(PETSc.ScalarType).eps
    Du.x.array[:] = eps

    timer1 = common.Timer("1st numba pass")
    timer1.start()
    evaluated_operands = evaluate_operands(F_external_operators)
    ((_, sigma_new, dp_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
    timer1.stop()

    timer2 = common.Timer("2nd numba pass")
    timer2.start()
    evaluated_operands = evaluate_operands(F_external_operators)
    ((_, sigma_new, dp_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)
    timer2.stop()

    u = fem.Function(V, name="displacement")
    du = fem.Function(V, name="Newton_correction")
    external_operator_problem = LinearProblem(J_replaced, F_replaced, Du, bcs=bcs)

    x_point = np.array([[R_i, 0, 0]])
    cells, points_on_process = find_cell_by_point(mesh, x_point)

    q_lim = 2.0 / np.sqrt(3.0) * np.log(R_e / R_i) * sigma_0
    num_increments = 20
    max_iterations, relative_tolerance = 200, 1e-8
    load_steps = (np.linspace(0, 1.1, num_increments, endpoint=True) ** 0.5)[1:]
    loadings = q_lim * load_steps

    for i, loading_v in enumerate(loadings):
        loading.value = loading_v
        external_operator_problem.assemble_vector()

        residual_0 = residual = external_operator_problem.b.norm()
        Du.x.array[:] = 0.0

        if MPI.COMM_WORLD.rank == 0:
            print(f"\nresidual , {residual} \n increment: {i+1!s}, load = {loading.value}")

        for iteration in range(0, max_iterations):
            if residual / residual_0 < relative_tolerance:
                break
            external_operator_problem.assemble_matrix()
            external_operator_problem.solve(du)
            du.x.scatter_forward()

            Du.vector.axpy(-1.0, du.vector)
            Du.x.scatter_forward()

            evaluated_operands = evaluate_operands(F_external_operators)

            # Implementation of an external operator may return several outputs and
            # not only its evaluation. For example, `C_tang_impl` returns a tuple of
            # Numpy-arrays with values of `C_tang`, `sigma` and `dp`.
            ((_, sigma_new, dp_new),) = evaluate_external_operators(J_external_operators, evaluated_operands)

            # In order to update the values of the external operator we may directly
            # access them and avoid the call of
            # `evaluate_external_operators(F_external_operators, evaluated_operands).`
            sigma.ref_coefficient.x.array[:] = sigma_new
            dp.x.array[:] = dp_new

            external_operator_problem.assemble_vector()
            residual = external_operator_problem.b.norm()

            if MPI.COMM_WORLD.rank == 0 and verbose:
                print(f"    it# {iteration} residual: {residual}")

        u.vector.axpy(1.0, Du.vector)
        u.x.scatter_forward()

        # Taking into account the history of loading
        p.vector.axpy(1.0, dp.vector)
        # skip scatter forward, p is not ghosted.
        # TODO: Why? What is the difference with lines above?
        sigma_n.x.array[:] = sigma.ref_coefficient.x.array
        # skip scatter forward, sigma is not ghosted.

    # TODO: Is there a more elegant way to extract the data?
    common.list_timings(MPI.COMM_WORLD, [common.TimingType.wall])

# via_numba()