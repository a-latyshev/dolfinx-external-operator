from collections.abc import Callable, Sequence

from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import numpy as np

import basix
from dolfinx.fem import Expression
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form
from dolfinx.fem.function import Function
from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points
from dolfinx.io import gmsh as gmshio


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
    mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=model_rank, gdim=2)
    mesh, cell_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
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


def interpolate_quadrature(ufl_expr, fem_func: Function) -> None:
    q_dim = fem_func.function_space._ufl_element.degree
    mesh = fem_func.ufl_function_space().mesh

    quadrature_points, _weights = basix.make_quadrature(basix.CellType.triangle, q_dim)
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    expr_expr = Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(mesh, cells)
    np.copyto(fem_func.x.array, expr_eval.reshape(-1))


def assemble_residual_with_callback(
    u: Function,
    F: Form,
    J: Form,
    bcs: Sequence[DirichletBC],
    external_callback: Callable,
    args_external_callback: Sequence,
    snes: PETSc.SNES,
    x: PETSc.Vec,
    b: PETSc.Vec,
) -> None:
    """Assemble the residual at ``x`` into the vector ``b`` with a callback to
    external functions.

    Prior to assembling the residual and after updating the solution ``u``, the
    function ``external_callback`` with input arguments ``args_external_callback``
    is called.

    A function conforming to the interface expected by ``SNES.setFunction`` can
    be created by fixing the first 5 arguments, e.g. (cf.
    ``dolfinx.fem.petsc.assemble_residual``):

    Example::

        snes = PETSc.SNES().create(mesh.comm)
        assemble_residual = functools.partial(
            dolfinx.fem.petsc.assemble_residual, u, F, J, bcs,
            external_callback, args_external_callback)
        snes.setFunction(assemble_residual, b)

    Args:
        u: Function tied to the solution vector within the residual and
           Jacobian.
        F: Form of the residual.
        J: Form of the Jacobian.
        bcs: List of Dirichlet boundary conditions to lift the residual.
        external_callback: A callback function to call prior to assembling the
                           residual.
        args_external_callback: Arguments to pass to the external callback
                                function.
        snes: The solver instance.
        x: The vector containing the point to evaluate the residual at.
        b: Vector to assemble the residual into.
    """
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    x.copy(u.x.petsc_vec)
    u.x.scatter_forward()

    # Call external functions, e.g. evaluation of external operators
    external_callback(*args_external_callback)

    with b.localForm() as b_local:
        b_local.set(0.0)
    assemble_vector(b, F)

    apply_lifting(b, [J], [bcs], [x], -1.0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs, x, -1.0)
