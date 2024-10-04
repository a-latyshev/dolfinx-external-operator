# Author: Andrey Latyshev
# Date: 2024
# This script is a part of the `dolfinx-external-operator` project, an extension
# of DOLFINx implementing the concept of  external operator

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
from solvers import LinearProblem
from utilities import build_cylinder_quarter, find_cell_by_point, interpolate_quadrature

import basix
import ufl
from dolfinx import fem


def plasticity_von_mises_pure_ufl(verbose=True):
    """"Implements von Mises plasticity analytically in UFL.

    It solves the 2D cylinder expansion problem. For the detailed problem
    formulation, please, take a look `demo_plasticity_von_mises.py` of the
    `dolfinx-external-operator` repository.

    This implementation outperforms a common one based on the interpolation of
    intermediate variables on each iteration of the Newton solver. For more
    details:
    https://bleyerj.github.io/comet-fenicsx/tours/nonlinear_problems/plasticity/plasticity.html
    """
    mesh, facet_tags, facet_tags_labels = build_cylinder_quarter()

    R_e, R_i = 1.3, 1.0  # external/internal radius

    # elastic parameters
    E = 70e3
    nu = 0.3
    lambda_ = E * nu / (1 + nu) / (1 - 2 * nu)
    mu_ = E / 2.0 / (1 + nu)

    sig0 = 250.0  # yield strength
    Et = E / 100.0  # tangent modulus
    H = E * Et / (E - Et)  # hardening modulus

    deg_u = 2
    deg_stress = 2
    V = fem.functionspace(mesh, ("Lagrange", deg_u, (2,)))
    W_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=deg_stress, value_shape=(4,))
    W = fem.functionspace(mesh, W_element)
    P_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=deg_stress, value_shape=())
    W0 = fem.functionspace(mesh, P_element)

    sig = fem.Function(W, name="Stress_vector")
    dp = fem.Function(W0, name="Cumulative_plastic_strain_increment")
    u = fem.Function(V, name="Total_displacement")
    du = fem.Function(V, name="Iteration_correction")
    Du = fem.Function(V, name="Current_increment")
    v_ = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = fem.Function(W0)

    bottom_facets = facet_tags.find(facet_tags_labels["Lx"])
    left_facets = facet_tags.find(facet_tags_labels["Ly"])

    bottom_dofs_y = fem.locate_dofs_topological(V.sub(1), mesh.topology.dim - 1, bottom_facets)
    left_dofs_x = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, left_facets)

    sym_bottom = fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), bottom_dofs_y, V.sub(1))
    sym_left = fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), left_dofs_x, V.sub(0))

    bcs = [sym_bottom, sym_left]

    n = ufl.FacetNormal(mesh)
    q_lim = float(2 / np.sqrt(3) * np.log(R_e / R_i) * sig0)
    loading = fem.Constant(mesh, PETSc.ScalarType(0.0 * q_lim))

    ds = ufl.Measure(
        "ds",
        domain=mesh,
        subdomain_data=facet_tags,
        metadata={"quadrature_degree": deg_stress, "quadrature_scheme": "default"},
    )
    dx = ufl.Measure(
        "dx",
        domain=mesh,
        metadata={"quadrature_degree": deg_stress, "quadrature_scheme": "default"},
    )

    def F_ext(v):
        """External force representing pressure acting on the inner wall of the cylinder."""
        return -loading * ufl.inner(n, v) * ds(3)

    def eps(v):
        e = ufl.sym(ufl.grad(v))
        return ufl.as_tensor([[e[0, 0], e[0, 1], 0], [e[0, 1], e[1, 1], 0], [0, 0, 0]])

    def sigma(eps_el):
        return lambda_ * ufl.tr(eps_el) * ufl.Identity(3) + 2 * mu_ * eps_el

    def as_3D_tensor(X):
        return ufl.as_tensor([[X[0], X[3], 0], [X[3], X[1], 0], [0, 0, X[2]]])

    def ppos(x):
        return (x + ufl.sqrt(x**2)) / 2.0

    def von_mises_expressions(Du, old_sig, old_p):
        """Computes analytical expressions of according to return-mapping.

        Returns:
            new_sig: A UFL-expression of the corrected stresses in Mandel-Voigt
            notation.
            dp: A UFL-expression of an increment of the cumulative plastic
            strain.
            deps_p: A UFL-expression of plastic strain.
        """
        sig_n = as_3D_tensor(old_sig)
        sig_elas = sig_n + sigma(eps(Du))
        s = ufl.dev(sig_elas)
        sig_eq = ufl.sqrt(3 / 2.0 * ufl.inner(s, s))
        f_elas = sig_eq - sig0 - H * old_p
        dp = ppos(f_elas) / (3 * mu_ + H)
        beta = 3 * mu_ * dp / sig_eq
        new_sig = sig_elas - beta * s
        deps_p = 3.0 / 2.0 * s / sig_eq * dp
        return ufl.as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), dp, deps_p

    sig_, dp_, deps_p = von_mises_expressions(Du, sig, p)

    residual = ufl.inner(as_3D_tensor(sig) + sigma(eps(Du) - deps_p), eps(v)) * dx - F_ext(v)
    J = ufl.derivative(ufl.inner(sigma(eps(Du) - deps_p), eps(v)) * dx, Du, v_)

    von_mises_problem = LinearProblem(J, -residual, Du, bcs)

    x_point = np.array([[R_i, 0, 0]])
    cells, points_on_proc = find_cell_by_point(mesh, x_point)

    TPV = np.finfo(PETSc.ScalarType).eps  # très petite value
    sig.x.array[:] = TPV

    Nitermax, tol = 200, 1e-8  # parameters of the manual Newton method
    Nincr = 20
    load_steps = (np.linspace(0, 1.1, Nincr, endpoint=True) ** 0.5)[1:]
    results = np.zeros((Nincr, 2))

    for i, t in enumerate(load_steps):
        loading.value = t * q_lim
        von_mises_problem.assemble_vector()

        nRes0 = von_mises_problem.b.norm()
        nRes = nRes0

        if MPI.COMM_WORLD.rank == 0 and verbose:
            print(f"\nIncrement#{i+1!s}: load = {t * q_lim:.3f}, Residual0 = {nRes0:.2e}")
        niter = 0

        while nRes / nRes0 > tol and niter < Nitermax:
            von_mises_problem.assemble_matrix()
            von_mises_problem.solve(du)

            Du.x.petsc_vec.axpy(1, du.x.petsc_vec)  # Du = Du + 1*du
            Du.x.scatter_forward()

            von_mises_problem.assemble_vector()

            nRes = von_mises_problem.b.norm()

            if MPI.COMM_WORLD.rank == 0 and verbose:
                print(f"\tit#{niter} Residual: {nRes:.2e}")
            niter += 1

        # Update main variables
        interpolate_quadrature(sig_, sig)
        interpolate_quadrature(dp_, dp)
        u.x.petsc_vec.axpy(1, Du.x.petsc_vec)  # u = u + 1*Du
        u.x.scatter_forward()
        p.x.array[:] = p.x.array + dp.x.array

        if len(points_on_proc) > 0:
            results[i + 1, :] = (u.eval(points_on_proc, cells)[0], t)

    return results