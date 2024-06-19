from mpi4py import MPI
from petsc4py import PETSc
import ufl
import basix
from dolfinx import fem

import numpy as np
import matplotlib.pyplot as plt

from solvers import LinearProblem
from utilities import build_cylinder_quarter, interpolate_quadrature, find_cell_by_point

def plasticity_von_mises_pure_ufl(verbose=True):
    mesh, facet_tags, facet_tags_labels = build_cylinder_quarter()

    R_e, R_i = 1.3, 1.   # external/internal radius

    # elastic parameters
    E = 70e3
    nu = 0.3
    lambda_ = E*nu/(1+nu)/(1-2*nu)
    mu_ = E/2./(1+nu)

    sig0 = 250.  # yield strength
    Et = E/100.  # tangent modulus
    H = E*Et/(E-Et)  # hardening modulus

    deg_u = 2
    deg_stress = 2
    V = fem.functionspace(mesh, ("Lagrange", deg_u, (2,)))
    W_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=deg_stress, value_shape=(4,))
    W = fem.functionspace(mesh, W_element)
    P_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=deg_stress, value_shape=())
    W0 = fem.functionspace(mesh, P_element)

    W2_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=deg_stress, value_shape=(2,))
    W2 = fem.functionspace(mesh, W2_element)

    # We = ufl.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=4, quad_scheme='default')
    # W0e = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
    # W2e = ufl.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=2, quad_scheme='default')
    # W = fem.FunctionSpace(mesh, We)
    # W0 = fem.FunctionSpace(mesh, W0e)
    # W2 = fem.FunctionSpace(mesh, W2e)

    sig = fem.Function(W, name="Stress_vector")
    p_W2 = fem.Function(W2, name="Cumulative_plastic_strain")
    dp = fem.Function(W0, name="Cumulative_plastic_strain_increment")
    u = fem.Function(V, name="Total_displacement")
    du = fem.Function(V, name="Iteration_correction")
    Du = fem.Function(V, name="Current_increment")
    v_ = ufl.TrialFunction(V)
    u_ = ufl.TestFunction(V)

    bottom_facets = facet_tags.find(facet_tags_labels["Lx"])
    left_facets = facet_tags.find(facet_tags_labels["Ly"])

    bottom_dofs_y = fem.locate_dofs_topological(V.sub(1), mesh.topology.dim-1, bottom_facets)
    left_dofs_x = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim-1, left_facets)

    sym_bottom = fem.dirichletbc(np.array(0.,dtype=PETSc.ScalarType), bottom_dofs_y, V.sub(1))
    sym_left = fem.dirichletbc(np.array(0.,dtype=PETSc.ScalarType), left_dofs_x, V.sub(0))

    bcs = [sym_bottom, sym_left]

    n = ufl.FacetNormal(mesh)
    q_lim = float(2/np.sqrt(3)*np.log(R_e/R_i)*sig0)
    loading = fem.Constant(mesh, PETSc.ScalarType(0.0 * q_lim))

    ds = ufl.Measure(
        "ds",
        domain=mesh,
        subdomain_data=facet_tags,
        metadata={"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
    )
    dx = ufl.Measure(
        "dx",
        domain=mesh,
        metadata={"quadrature_degree": deg_stress, "quadrature_scheme": "default"},
    )

    def F_ext(v):
        """External force representing pressure acting on the inner wall of the cylinder."""
        return -loading * ufl.inner(n, v)*ds(3)

    def eps(v):
        e = ufl.sym(ufl.grad(v))
        return ufl.as_tensor([[e[0, 0], e[0, 1], 0],
                            [e[0, 1], e[1, 1], 0],
                            [0, 0, 0]])

    def sigma(eps_el):
        return lambda_*ufl.tr(eps_el)*ufl.Identity(3) + 2*mu_*eps_el

    def as_3D_tensor(X):
        return ufl.as_tensor([[X[0], X[3], 0],
                            [X[3], X[1], 0],
                            [0, 0, X[2]]])

    SQRT2 = np.sqrt(2.)
    def new_eps(v):
        grad_v = ufl.grad(v)
        return ufl.as_vector([grad_v[0,0], grad_v[1,1], 0, SQRT2 * 0.5*(grad_v[0,1] + grad_v[1,0])])

    ppos = lambda x: (x + ufl.sqrt(x**2))/2.

    def proj_sig_SNES(Du, old_sig, old_p):
        sig_n = as_3D_tensor(old_sig)
        sig_elas = sig_n + sigma(eps(Du))
        s = ufl.dev(sig_elas)
        sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
        f_elas = sig_eq - sig0 - H*old_p
        dp = ppos(f_elas)/(3*mu_ + H)
        # beta = ufl.conditional(f_elas > 0, 3*mu_*dp/sig_eq, 0)
        beta = 3*mu_*dp/sig_eq
        new_sig = sig_elas-beta*s
        return ufl.as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
            dp

    def deps_p(deps, old_sig, old_p):
        sig_n = as_3D_tensor(old_sig)
        sig_elas = sig_n + sigma(deps)
        s = ufl.dev(sig_elas)
        sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
        f_elas = sig_eq - sig0 - H*old_p
        dp = ppos(f_elas)/(3*mu_+H)
        # dp_sig_eq = ufl.conditional(f_elas > 0, f_elas/(3*mu_ + H)/sig_eq, 0) # Note: sig_eq is equal to 0 on the first iteration
        # dp = ppos(f_elas)/(3*mu+H) # this approach doesn't work with ufl.derivate
        return 3./2. * s/sig_eq * dp
        # return 3./2. * dp_sig_eq * s
    TPV = np.finfo(PETSc.ScalarType).eps # très petite value 
    # Du.x.array[:] = TPV # For initialization 
    sig.x.array[:] = TPV
    # def deps_p(deps, old_sig, old_p):
    #     sig_n = as_3D_tensor(old_sig)
    #     sig_elas = sig_n + sigma(deps)
    #     s = ufl.dev(sig_elas)
    #     sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
    #     f_elas = sig_eq - sig0 - H*old_p
    #     # dp = ppos(f_elas)/(3*mu_+H)
    #     dp_sig_eq = ufl.conditional(f_elas > 0, f_elas/(3*mu_ + H)/sig_eq, 0) # Note: sig_eq is equal to 0 on the first iteration
    #     # dp = ppos(f_elas)/(3*mu+H) # this approach doesn't work with ufl.derivate
    #     # return 3./2. * s/sig_eq * old_p
    #     return 3./2. * dp_sig_eq * s
    p = p_W2[0]
    p_values = p_W2.x.array.reshape((-1, 2))[:,0]

    # print(p_values.shape, dp.x.array.shape)
    residual = ufl.inner(as_3D_tensor(sig) + sigma(eps(Du) - deps_p(eps(Du), sig, p)), eps(u_))*dx - F_ext(u_)

    # new_deps_p = deps_p(eps(Du), sig, p)
    # new_sig = sigma(new_deps_p)
    # residual = ufl.inner(as_3D_tensor(sig*p), eps(u_))*dx -loading * ufl.inner(n, u_)*ds(3)
    # print(p.ufl_shape, new_deps_p.ufl_shape)
    # fem.form(residual)
    J = ufl.derivative(ufl.inner(sigma(eps(Du) - deps_p(eps(Du), sig, p)), eps(u_))*dx, Du, v_)

    my_problem = LinearProblem(J, -residual, Du, bcs)

    # Defining a cell containing (Ri, 0) point, where we calculate a value of u
    # It is required to run this program via MPI in order to capture the process, to which this point is attached 

    x_point = np.array([[R_i, 0, 0]])
    cells, points_on_proc = find_cell_by_point(mesh, x_point)

    Nitermax, tol = 200, 1e-8  # parameters of the manual Newton method
    Nincr = 20
    load_steps = np.linspace(0, 1.1, Nincr+1)[1:]**0.5
    results = np.zeros((Nincr+1, 2))

    sig_, dp_ = proj_sig_SNES(Du, sig, p)

    # time_monitor = {}
    # time_monitor["solve_lin_problem"] = 0
    # time_monitor["matrix_assembling"] = 0
    # time_monitor["after_Newton_update"] = 0
    # time_monitor["compilation_overhead"] = 0
    # time_monitor["total_iterations"] = 0

    start_run_time = MPI.Wtime()

    for (i, t) in enumerate(load_steps):
        loading.value = t * q_lim
        my_problem.assemble_vector()

        nRes0 = my_problem.b.norm() 
        nRes = nRes0
        # Du.x.array[:] = 0

        if MPI.COMM_WORLD.rank == 0 and verbose:
            print(f"\nIncrement#{str(i+1)}: load = {t * q_lim:.3f}, Residual0 = {nRes0:.2e}")
        niter = 0

        while nRes/nRes0 > tol and niter < Nitermax:
            start = MPI.Wtime()
            my_problem.assemble_matrix()
            end = MPI.Wtime()
            # time_monitor["matrix_assembling"] += end - start
            start = MPI.Wtime()
            my_problem.solve(du)
            end = MPI.Wtime()
            # time_monitor["solve_lin_problem"] += end - start

            Du.vector.axpy(1, du.vector) # Du = Du + 1*du
            Du.x.scatter_forward() 

            my_problem.assemble_vector()

            nRes = my_problem.b.norm() 

            if MPI.COMM_WORLD.rank == 0 and verbose:
                print(f"\tit#{niter} Residual: {nRes:.2e}")
            niter += 1

        # time_monitor["total_iterations"] += niter

        start = MPI.Wtime()
        interpolate_quadrature(sig_, sig)
        interpolate_quadrature(dp_, dp)

        u.vector.axpy(1, Du.vector) # u = u + 1*Du
        u.x.scatter_forward()

        # p.vector.axpy(1, dp.vector)
        # p.x.scatter_forward()
        np.copyto(p_values, p_values + dp.x.array)
        end = MPI.Wtime()
        # time_monitor["after_Newton_update"] += end - start

        if len(points_on_proc) > 0:

            results[i+1, :] = (u.eval(points_on_proc, cells)[0], t)
    end_run_time = MPI.Wtime()

    plt.plot(results[:,0], results[:,1], "-o")
    plt.savefig('test.png')
    # time_monitor["total_time"] = end_run_time - start_run_time
    # print(f'rank#{MPI.COMM_WORLD.rank}: Time = {total_time:.3f} (s)')
    return results