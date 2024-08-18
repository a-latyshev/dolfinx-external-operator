from mpi4py import MPI
from petsc4py import PETSc

import jax
import jax.numpy as jnp
import gmsh
import numpy as np
import pyvista

import basix
import dolfinx.plot as plot
from dolfinx.fem import Expression, Function
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points
from dolfinx.io import gmshio


def build_cylinder_quarter(lc=0.3, R_e=1.3, R_i=1.0):
    # Source: https://newfrac.github.io/fenicsx-fracture/notebooks/plasticity/plasticity.html

    # mesh parameters
    gdim = 2
    verbosity = 0
    model_rank = 0

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", verbosity)

    facet_tags_labels = {"Lx": 1, "Ly": 2, "inner": 3, "outer": 4}

    cell_tags_map = {"all": 20}

    if MPI.COMM_WORLD.rank == model_rank:
        model = gmsh.model()
        model.add("quarter_cylinder")
        model.setCurrent("quarter_cylinder")
        # Create the points
        pix = model.occ.addPoint(R_i, 0.0, 0, lc)
        pex = model.occ.addPoint(R_e, 0, 0, lc)
        piy = model.occ.addPoint(0.0, R_i, 0, lc)
        pey = model.occ.addPoint(0.0, R_e, 0, lc)
        center = model.occ.addPoint(0.0, 0.0, 0, lc)
        # Create the lines
        lx = model.occ.addLine(pix, pex, tag=facet_tags_labels["Lx"])
        lout = model.occ.addCircleArc(pex, center, pey, tag=facet_tags_labels["outer"])
        ly = model.occ.addLine(pey, piy, tag=facet_tags_labels["Ly"])
        lin = model.occ.addCircleArc(piy, center, pix, tag=facet_tags_labels["inner"])
        # Create the surface
        cloop1 = model.occ.addCurveLoop([lx, lout, ly, lin])
        _ = model.occ.addPlaneSurface([cloop1], tag=cell_tags_map["all"])
        model.occ.synchronize()
        # Assign mesh and facet tags
        surface_entities = [entity[1] for entity in model.getEntities(2)]
        model.addPhysicalGroup(2, surface_entities, tag=cell_tags_map["all"])
        model.setPhysicalName(2, 2, "Quart_cylinder surface")
        for key, value in facet_tags_labels.items():
            # 1 : it is the dimension of the object (here a curve)
            model.addPhysicalGroup(1, [value], tag=value)
            model.setPhysicalName(1, value, key)
        # Finalize mesh
        model.occ.synchronize()
        model.mesh.generate(gdim)

    # NOTE: Do not forget to check the leaks produced by the following line of code
    # [WARNING] yaksa: 2 leaked handle pool objects
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=model_rank, gdim=2)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.name = "quarter_cylinder"
    cell_tags.name = f"{mesh.name}_cells"
    facet_tags.name = f"{mesh.name}_facets"

    gmsh.finalize()

    return mesh, facet_tags, facet_tags_labels


def find_cell_by_point(mesh, point):
    cells = []
    points_on_proc = []
    tree = bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = compute_collisions_points(tree, point)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, point)
    for i, point in enumerate(point):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    return cells, points_on_proc


def plot_scalar_field(field, verbose=False, to_show=True):
    """
    Plot a scalar field with pyvista
    """
    field_name = field.name
    domain = field.function_space.mesh
    plotter = pyvista.Plotter(title=field_name, window_size=[200, 300])
    topology, cell_types, x = plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    grid.point_data[field_name] = field.x.array
    grid.set_active_scalars(field_name)
    if verbose:
        plotter.add_text(field_name, font_size=11)
    plotter.add_mesh(grid, show_edges=False, show_scalar_bar=verbose)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN and to_show:
        plotter.show()
    plotter.camera.tight()
    image = plotter.screenshot(None, transparent_background=True, return_img=True)
    return image


def interpolate_quadrature(ufl_expr, fem_func: Function):
    q_dim = fem_func.function_space._ufl_element.degree
    mesh = fem_func.ufl_function_space().mesh

    quadrature_points, weights = basix.make_quadrature(basix.CellType.triangle, q_dim)
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    expr_expr = Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(mesh, cells)
    np.copyto(fem_func.x.array, expr_eval.reshape(-1))


def Mohr_Coulomb_yield_criterion(phi, c, E, nu):
    """"Returns return-mapping procedure for standard Mohr-Coulomb yield criterion."""
    stress_dim = 4
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C_elas = np.array(
        [
            [lmbda + 2 * mu, lmbda, lmbda, 0],
            [lmbda, lmbda + 2 * mu, lmbda, 0],
            [lmbda, lmbda, lmbda + 2 * mu, 0],
            [0, 0, 0, 2 * mu],
        ],
        dtype=PETSc.ScalarType,
    )
    ZERO_VECTOR = np.zeros(stress_dim, dtype=PETSc.ScalarType)

    dev = np.array(
        [
            [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 0.0],
            [-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 0.0],
            [-1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=PETSc.ScalarType,
    )
    tr = np.array([1.0, 1.0, 1.0, 0.0], dtype=PETSc.ScalarType)

    def J3(s):
        return s[2] * (s[0] * s[1] - s[3] * s[3] / 2.0)

    def J2(s):
        return 0.5 * jnp.vdot(s, s)

    def theta(s):
        J2_ = J2(s)
        arg = -(3.0 * np.sqrt(3.0) * J3(s)) / (2.0 * jnp.sqrt(J2_ * J2_ * J2_))
        arg = jnp.clip(arg, -1.0, 1.0)
        theta = 1.0 / 3.0 * jnp.arcsin(arg)
        return theta

    def f(sigma_local):
        s = dev @ sigma_local
        I1 = tr @ sigma_local
        theta_ = theta(s)
        K_ = jnp.cos(theta_) - (1.0 / np.sqrt(3.0)) * np.sin(phi) * jnp.sin(theta_)
        return (I1 / 3.0 * np.sin(phi) + jnp.sqrt(J2(s)) * K_ - c * np.cos(phi))

    dfdsigma = jax.jacfwd(f)

    def deps_p(sigma_local, dlambda, deps_local, sigma_n_local):
        sigma_elas_local = sigma_n_local + C_elas @ deps_local
        yielding = f(sigma_elas_local)

        def deps_p_elastic(sigma_local, dlambda):
            return ZERO_VECTOR

        def deps_p_plastic(sigma_local, dlambda):
            return dlambda * dfdsigma(sigma_local)

        return jax.lax.cond(yielding <= 0.0, deps_p_elastic, deps_p_plastic, sigma_local, dlambda)


    def r_g(sigma_local, dlambda, deps_local, sigma_n_local):
        deps_p_local = deps_p(sigma_local, dlambda, deps_local, sigma_n_local)
        return sigma_local - sigma_n_local - C_elas @ (deps_local - deps_p_local)


    def r_f(sigma_local, dlambda, deps_local, sigma_n_local):
        sigma_elas_local = sigma_n_local + C_elas @ deps_local
        yielding = f(sigma_elas_local)

        def r_f_elastic(sigma_local, dlambda):
            return dlambda

        def r_f_plastic(sigma_local, dlambda):
            return f(sigma_local)

        return jax.lax.cond(yielding <= 0.0, r_f_elastic, r_f_plastic, sigma_local, dlambda)


    def r(x_local, deps_local, sigma_n_local):
        sigma_local = x_local[:stress_dim]
        dlambda_local = x_local[-1]

        res_g = r_g(sigma_local, dlambda_local, deps_local, sigma_n_local)
        res_f = r_f(sigma_local, dlambda_local, deps_local, sigma_n_local)

        res = jnp.c_["0,1,-1", res_g, res_f]  # concatenates an array and a scalar
        return res


    drdx = jax.jacfwd(r)
    ZERO_SCALAR = np.array([0.0])
    Nitermax, tol = 200, 1e-10

    def return_mapping(deps_local, sigma_n_local):
        niter = 0

        dlambda = ZERO_SCALAR
        sigma_local = sigma_n_local
        x_local = jnp.concatenate([sigma_local, dlambda])

        res = r(x_local, deps_local, sigma_n_local)
        norm_res0 = jnp.linalg.norm(res)

        def cond_fun(state):
            norm_res, niter, _ = state
            return jnp.logical_and(norm_res / norm_res0 > tol, niter < Nitermax)

        def body_fun(state):
            norm_res, niter, history = state

            x_local, deps_local, sigma_n_local, res = history

            j = drdx(x_local, deps_local, sigma_n_local)
            j_inv_vp = jnp.linalg.solve(j, -res)
            x_local = x_local + j_inv_vp

            res = r(x_local, deps_local, sigma_n_local)
            norm_res = jnp.linalg.norm(res)
            history = x_local, deps_local, sigma_n_local, res

            niter += 1

            return (norm_res, niter, history)

        history = (x_local, deps_local, sigma_n_local, res)

        norm_res, niter_total, x_local = jax.lax.while_loop(cond_fun, body_fun, (norm_res0, niter, history))

        sigma_local = x_local[0][:stress_dim]
        dlambda = x_local[0][-1]
        sigma_elas_local = C_elas @ deps_local
        yielding = f(sigma_n_local + sigma_elas_local)

        return sigma_local, (sigma_local, niter_total, yielding, norm_res, dlambda)

    return return_mapping
