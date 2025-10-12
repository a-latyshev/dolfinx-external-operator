from mpi4py import MPI

import gmsh
import numpy as np

# import pyvista
import basix

# import dolfinx.plot as plot
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

    quadrature_points, weights = basix.make_quadrature(basix.CellType.triangle, q_dim)
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    expr_expr = Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(mesh, cells)
    np.copyto(fem_func.x.array, expr_eval.reshape(-1))


def build_square_with_elliptic_holes(lc=0.05, L=1.0, hole1_center=(0.375, 0.25), hole1_radii=(0.25, 0.125), hole2_center=(0.75, 0.625), hole2_radii=(0.25, 0.125)):
    """
    Build a square domain with two asymmetric elliptical holes using GMSH.

    Args:
        lc: Mesh size parameter.
        L: Length of the square domain.
        hole1_center: Center of the first elliptical hole (x, y).
        hole1_radii: Radii of the first elliptical hole (rx, ry).
        hole2_center: Center of the second elliptical hole (x, y).
        hole2_radii: Radii of the second elliptical hole (rx, ry).

    Returns:
        mesh, cell_tags, facet_tags: Generated mesh and tags.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)

    facet_tags_labels = {"bottom": 1, "top": 2, "right": 3, "left": 4}

    if MPI.COMM_WORLD.rank == 0:
        model = gmsh.model()
        model.add("square_with_elliptic_holes")

        # Create the square domain
        p1 = model.occ.addPoint(0, 0, 0, lc)
        p2 = model.occ.addPoint(L, 0, 0, lc)
        p3 = model.occ.addPoint(L, L, 0, lc)
        p4 = model.occ.addPoint(0, L, 0, lc)
        square_loop = model.occ.addCurveLoop([
            model.occ.addLine(p1, p2, tag=facet_tags_labels["bottom"]),
            model.occ.addLine(p2, p3, tag=facet_tags_labels["right"]),
            model.occ.addLine(p3, p4, tag=facet_tags_labels["top"]),
            model.occ.addLine(p4, p1, tag=facet_tags_labels["left"])
        ])

        # Create the first elliptical hole
        hole1 = model.occ.addEllipse(hole1_center[0], hole1_center[1], 0, hole1_radii[0], hole1_radii[1])
        hole1_loop = model.occ.addCurveLoop([hole1])

        # Create the second elliptical hole
        hole2 = model.occ.addEllipse(hole2_center[0], hole2_center[1], 0, hole2_radii[0], hole2_radii[1], zAxis=[0, 0, 1], xAxis=[0, 1, 0])
        hole2_loop = model.occ.addCurveLoop([hole2])

        # Find all points in the model and assign the same mesh size to all of them
        all_points = model.occ.getEntities(0)
        model.occ.mesh.setSize(all_points, lc)

        # Create the surface with holes
        surface = model.occ.addPlaneSurface([square_loop, hole1_loop, hole2_loop])
        model.occ.synchronize()

        # Assign physical groups
        model.addPhysicalGroup(2, [surface], tag=1)
        model.setPhysicalName(2, 1, "Square_with_elliptic_holes")
        for key, value in facet_tags_labels.items():
            # 1 : it is the dimension of the object (here a curve)
            model.addPhysicalGroup(1, [value], tag=value)
            model.setPhysicalName(1, value, key)

        # Generate the mesh
        model.mesh.generate(2)
        gmsh.write("square_with_elliptic_holes.geo_unrolled")
        gmsh.write("square_with_elliptic_holes.msh")
    # Convert to dolfinx mesh
    mesh_data = gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2)
    mesh, cell_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    gmsh.finalize()

    return mesh, facet_tags, facet_tags_labels