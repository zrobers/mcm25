import pygmsh
import gmsh
import numpy as np
import plotly.graph_objects as go

# FEniCS/dolfinx-related imports
import dolfinx
from dolfinx.io import XDMFFile
# Use model_to_mesh instead of gmsh_to_mesh
from dolfinx.io import gmshio

# For parallel usage:
from mpi4py import MPI

def create_stair_mesh_tet(width, depth, height, dx):
    """
    Creates a 3D hexahedral-like mesh for a stair step domain:
      - width  in X-direction (1.0 m  = 100 cm)
      - depth  in Y-direction (0.3 m  = 30  cm)
      - height in Z-direction (0.1 m  = 10  cm)
    with approximate cell size dx (0.01 m = 1 cm).

    Returns:
        mesh (dolfinx.Mesh),
        points (np.ndarray),
        cells_arr (np.ndarray)  # local to rank=0
    """
    # Initialize gmsh
    # gmsh.initialize()
    # gmsh.model.add("StairStep")


    # Domain corners:
    x0, y0, z0 = 0.0, 0.0, 0.0
    x1, y1, z1 = width, depth, height

    # Number of divisions along each direction
    nx = int((x1 - x0) / dx)
    ny = int((y1 - y0) / dx)
    nz = int((z1 - z0) / dx)
    
    # with pygmsh.geo.Geometry() as geom:
    #     box = geom.add_box(x0, x1, y0, y1, z0, z1, dx, True)
    #     gmsh.option.setNumber("Mesh.RecombineAll", 1)
    #     geom.generate_mesh(dim=3)
    #     gmsh.model.geo.synchronize()
    #     gmsh.model.mesh.generate(3)
    #     gmsh.write("debug_box.msh")

    with pygmsh.geo.Geometry() as geom:
        
        gmsh.option.setNumber("Mesh.Algorithm", 8)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", dx)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", dx)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        
        # Create a constant mesh size field
        mesh_size = dx  # Set your desired mesh size
        # field = gmsh.model.mesh.field.add("Constant")
        # gmsh.model.mesh.field.setNumber(field, "Size", mesh_size)
        
        # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        
        # Define corner points
        p0 = geom.add_point([x0, y0, z0], dx)
        p1 = geom.add_point([x1, y0, z0], dx)
        p2 = geom.add_point([x1, y1, z0], dx)
        p3 = geom.add_point([x0, y1, z0], dx)
        p4 = geom.add_point([x0, y0, z1], dx)
        p5 = geom.add_point([x1, y0, z1], dx)
        p6 = geom.add_point([x1, y1, z1], dx)
        p7 = geom.add_point([x0, y1, z1], dx)

        # Create lines
        l0 = geom.add_line(p0, p1)
        l1 = geom.add_line(p1, p2)
        l2 = geom.add_line(p2, p3)
        l3 = geom.add_line(p3, p0)

        l4 = geom.add_line(p4, p5)
        l5 = geom.add_line(p5, p6)
        l6 = geom.add_line(p6, p7)
        l7 = geom.add_line(p7, p4)

        l8  = geom.add_line(p0, p4)
        l9  = geom.add_line(p1, p5)
        l10 = geom.add_line(p2, p6)
        l11 = geom.add_line(p3, p7)

        # Surfaces
        bottom_loop = geom.add_curve_loop([-l0, -l3, -l2, -l1])
        top_loop    = geom.add_curve_loop([l4, l5, l6, l7])
        side1_loop  = geom.add_curve_loop([l0, l9, -l4, -l8])
        side2_loop  = geom.add_curve_loop([l1, l10, -l5, -l9])
        side3_loop  = geom.add_curve_loop([l2, l11, -l6, -l10])
        side4_loop  = geom.add_curve_loop([l3, l8, -l7, -l11])

        s_bottom = geom.add_plane_surface(bottom_loop)
        s_top    = geom.add_plane_surface(top_loop)
        s_side1  = geom.add_plane_surface(side1_loop)
        s_side2  = geom.add_plane_surface(side2_loop)
        s_side3  = geom.add_plane_surface(side3_loop)
        s_side4  = geom.add_plane_surface(side4_loop)

        surface_loop = geom.add_surface_loop([s_bottom, s_side1, s_side2, s_side3, s_side4, s_top])
        volume       = geom.add_volume(surface_loop)
        
        
        
        # Transfinite curves (for structured approach)
        geom.set_transfinite_curve(l0, nx+1, "Progression", 1.0)
        geom.set_transfinite_curve(l1, ny+1, "Progression", 1.0)
        geom.set_transfinite_curve(l2, nx+1, "Progression", 1.0)
        geom.set_transfinite_curve(l3, ny+1, "Progression", 1.0)
        geom.set_transfinite_curve(l4, nx+1, "Progression", 1.0)
        geom.set_transfinite_curve(l5, ny+1, "Progression", 1.0)
        geom.set_transfinite_curve(l6, nx+1, "Progression", 1.0)
        geom.set_transfinite_curve(l7, ny+1, "Progression", 1.0)
        geom.set_transfinite_curve(l8, nz+1, "Progression", 1.0)
        geom.set_transfinite_curve(l9, nz+1, "Progression", 1.0)
        geom.set_transfinite_curve(l10, nz+1, "Progression", 1.0)
        geom.set_transfinite_curve(l11, nz+1, "Progression", 1.0)
        

        # Bottom (p0, p1, p2, p3)
        geom.set_transfinite_surface(s_bottom, "Corners", [p0, p3, p2, p1])
        # Top (p4, p5, p6, p7)
        geom.set_transfinite_surface(s_top,    "Corners", [p4, p5, p6, p7])
        # side1 (p0, p1, p5, p4)
        geom.set_transfinite_surface(s_side1,  "Corners", [p0, p1, p5, p4])
        # side2 (p1, p2, p6, p5)
        geom.set_transfinite_surface(s_side2,  "Corners", [p1, p2, p6, p5])
        # side3 (p2, p3, p7, p6)
        geom.set_transfinite_surface(s_side3,  "Corners", [p2, p3, p7, p6])
        # side4 (p3, p0, p4, p7)
        geom.set_transfinite_surface(s_side4,  "Corners", [p3, p0, p4, p7])

        geom.set_transfinite_volume(volume, [p0, p1, p2, p3, p4, p5, p6, p7])

        # geom.add_physical(volume, "StairStep")

        # Attempt Recombine for hex
        # gmsh.option.setNumber("Mesh.RecombineAll", 1)
        # Or set_recombine(3, surfaceID) if needed

        # Generate
        geom.generate_mesh(dim=3)
    
        v_entities = gmsh.model.getEntities(dim=3)
        print("3D volume entities:", v_entities)
                
        # Synchronize
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        
        gmsh.model.addPhysicalGroup(3, [1], 1)
        gmsh.model.setPhysicalName(3, 1, "Volume1")
        
        # types_3d, tags_3d, node_ids_3d = gmsh.model.mesh.getElements(dim=3)

        # vol_entities = gmsh.model.getEntities(dim=3)
        # print("3D volume entities:", vol_entities)
        
        # print("3D element types:", types_3d)
        # print("Number of 3D element types:", len(types_3d))
        # if len(types_3d) > 0:
        #     print("First element type:", types_3d[0])
        #     print("Number of elements in that type:", len(tags_3d[0]))

        # result = gmshio.extract_topology_and_markers(gmsh.model)

        # print("The result: ", result)

        gmsh.write("debug_mesh.msh")  

        # Convert to dolfinx mesh (rank 0 does the reading)
        mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=3)

    # gmsh.finalize()

    # We only do the below steps on rank=0
    if mesh.comm.rank == 0:
        # Access geometry/coordinates
        x = mesh.geometry.x  # shape (#nodes, 3)
        # We must ensure we have connectivity from cells -> points
        conn = mesh.topology.connectivity(mesh.topology.dim, 0)
        # get adjacency        
        node_array = conn.array
        offsets = conn.offsets
        # If single shape => a known number of nodes per cell
        known_num_nodes = 4
        cells_arr = node_array.reshape(-1, known_num_nodes)
    else:
        x = np.zeros((0,3), dtype=np.float64)
        cells_arr = np.zeros((0,8), dtype=np.int32)

    return mesh, x, cells_arr

if __name__ == "__main__":
        
    # Desired step geometry: 1m x 0.3m x 0.1m
    # with 1cm resolution => dx=0.01
    mesh, points, cells_tet = create_stair_mesh_tet(width=1, depth=1, height=1, dx=0.5)

    # Only rank 0 writes the XDMF and makes the Plotly figure
    if mesh.comm.rank == 0:
        # Save XDMF
        with XDMFFile(mesh.comm, "stair_step_mesh.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
        print(f"Mesh saved to 'stair_step_mesh.xdmf'.")
        print(f"Number of points in mesh: {len(points)}")
        print(f"Number of cells: {len(cells_tet)}")

        # Create a wireframe with Plotly
        # We do the same "edge" approach as before
        if len(cells_tet) > 3000:
            # random subset to avoid huge plot
            import random
            idx_subset = random.sample(range(len(cells_tet)), 200)
        else:
            idx_subset = range(len(cells_tet))

        edge_x = []
        edge_y = []
        edge_z = []

        def add_line(p, q):
            edge_x.append(p[0]); edge_y.append(p[1]); edge_z.append(p[2])
            edge_x.append(q[0]); edge_y.append(q[1]); edge_z.append(q[2])
            edge_x.append(None); edge_y.append(None); edge_z.append(None)

        # typical 4-node hex edges
        tet_edges = [
            (0,1), (1,2), (2,0),  # base face edges
            (0,3), (1,3), (2,3)   # edges to the apex
        ]

        for c_index in idx_subset:
            cell_nodes = cells_tet[c_index]
            for (n1, n2) in tet_edges:
                p = points[cell_nodes[n1]]
                q = points[cell_nodes[n2]]
                add_line(p, q)

        fig = go.Figure(data=[
            go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='blue', width=2),
                name='HexEdges'
            )
        ])
        fig.update_layout(
            scene=dict(aspectmode='data'),
            title="Stair Step Mesh (Wireframe Subset)"
        )
        fig.show()


