from mpi4py import MPI

import gmsh

from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points
from dolfinx.io import gmshio


def build_cylinder_quarter(lc=0.3, R_e=1.3, R_i=1.0):
    # Source: https://newfrac.github.io/fenicsx-fracture/notebooks/plasticity/plasticity.html

    # mesh parameters
    gdim = 2
    lc = 0.3
    verbosity = 0
    model_rank = 0
    gmsh.initialize()

    facet_tags_labels = {"Lx": 1, "Ly": 2, "inner": 3, "outer": 4}

    cell_tags_map = {"all": 20}
    if MPI.COMM_WORLD.rank == model_rank:
        model = gmsh.model()
        model.add("Quart_cylinder")
        model.setCurrent("Quart_cylinder")
        # Create the points
        pix = model.occ.addPoint(R_i, 0.0, 0, lc)
        pex = model.occ.addPoint(R_e, 0, 0, lc)
        piy = model.occ.addPoint(0.0, R_i, 0, lc)
        pey = model.occ.addPoint(0.0, R_e, 0, lc)
        center = model.occ.addPoint(0.0, 0.0, 0, lc)
        # Create the lines
        lx = model.occ.addLine(pix, pex, tag=facet_tags_labels["Lx"])
        lout = model.occ.addCircleArc(
            pex, center, pey, tag=facet_tags_labels["outer"])
        ly = model.occ.addLine(pey, piy, tag=facet_tags_labels["Ly"])
        lin = model.occ.addCircleArc(
            piy, center, pix, tag=facet_tags_labels["inner"])
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
        gmsh.option.setNumber("General.Verbosity", verbosity)
        model.mesh.generate(gdim)

    # NOTE: Do not forget to check the leaks produced by the following line of code
    # [WARNING] yaksa: 2 leaked handle pool objects
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0.0, gdim=2)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.name = "quarter_cylinder"
    cell_tags.name = f"{mesh.name}_cells"
    facet_tags.name = f"{mesh.name}_facets"

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
