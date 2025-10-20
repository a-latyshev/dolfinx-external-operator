from mpi4py import MPI

import gmsh
import numpy as np

# import pyvista
import basix

# import dolfinx.plot as plot
from dolfinx.fem import Expression, Function
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


# def plot_scalar_field(field, verbose=False, to_show=True):
#     """
#     Plot a scalar field with pyvista
#     """
#     field_name = field.name
#     domain = field.function_space.mesh
#     plotter = pyvista.Plotter(title=field_name, window_size=[200, 300])
#     topology, cell_types, x = plot.vtk_mesh(domain)
#     grid = pyvista.UnstructuredGrid(topology, cell_types, x)
#     grid.point_data[field_name] = field.x.array
#     grid.set_active_scalars(field_name)
#     if verbose:
#         plotter.add_text(field_name, font_size=11)
#     plotter.add_mesh(grid, show_edges=False, show_scalar_bar=verbose)
#     plotter.view_xy()
#     if not pyvista.OFF_SCREEN and to_show:
#         plotter.show()
#     plotter.camera.tight()
#     image = plotter.screenshot(None, transparent_background=True, return_img=True)
#     return image


def interpolate_quadrature(ufl_expr, fem_func: Function):
    q_dim = fem_func.function_space._ufl_element.degree
    mesh = fem_func.ufl_function_space().mesh

    quadrature_points, _weights = basix.make_quadrature(basix.CellType.triangle, q_dim)
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    expr_expr = Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(mesh, cells)
    np.copyto(fem_func.x.array, expr_eval.reshape(-1))
