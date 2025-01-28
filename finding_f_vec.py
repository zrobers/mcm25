import numpy as np
from mpi4py import MPI
import pyvista as pv


# Dolfinx post-2023
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from dolfinx.geometry import BoundingBoxTree

def top_marker(x, z_top=0.1, tol=1e-9):
    """
    Boolean function marking top boundary facets if all corner nodes 
    have z ~ z_top within a tolerance.
    x.shape = (3, N) for N points in 3D.
    """
    return np.isclose(x[2], z_top, atol=tol)

def compute_area_normal_rect(coords: np.ndarray):
    """
    Compute area and outward normal for a 'rectangular' face 
    with corner ordering (0->1->2->3). 
    coords shape = (4,3).
    We'll do the cross product of two edges:
      v1 = coords[1] - coords[0]
      v2 = coords[3] - coords[0]
    area = norm(cross(v1, v2))
    normal = cross(...) / area

    If the face is not exactly planar, or corner ordering is not 
    strictly rectangular, adapt as needed.
    """
    if coords.shape != (4,3):
        raise ValueError("This function requires exactly 4 corner nodes in consistent rectangular order.")
    v1 = coords[1] - coords[0]
    v2 = coords[3] - coords[0]
    cross_vec = np.cross(v1, v2)
    area = np.linalg.norm(cross_vec)
    if area < 1e-14:
        # degenerate or zero area
        return (0.0, np.array([0,0,0], dtype=np.float64))
    normal = cross_vec / area
    return (area, normal)

def build_load_vector(mesh, pressure_value=1e4, z_top=0.1):
    """
    1) Identify top boundary facets at z ~ z_top
    2) For each facet, compute area & normal via compute_area_normal_rect
    3) Distribute uniform load to corner nodes
    Returns a numpy array f_global of shape (#nodes*3).
    """

    # We'll define 3D dofs: each node has 3 displacement DOFs
    num_nodes = mesh.geometry.x.shape[0]
    f_global = np.zeros(num_nodes*3, dtype=np.float64)

    # 1) locate top facets
    # dimension=2 for boundary faces in a 3D mesh
    top_facets = locate_entities_boundary(mesh, dim=2, marker=lambda x: top_marker(x, z_top=z_top))

    # 2) build or ensure connectivity from face->node
    c = mesh.topology.connectivity(2, 0)
    if c is None:
        mesh.topology.create_connectivity(2, 0)
        c = mesh.topology.connectivity(2, 0)

    offsets = c.offsets
    node_array = c.array
    coords_all = mesh.geometry.x  # shape (#nodes, 3)

    for fidx in top_facets:
        start = offsets[fidx]
        end   = offsets[fidx+1]
        corner_nodes = node_array[start:end]  # e.g. 4 corner node indices
        if len(corner_nodes) != 4:
            # or skip if not a 4-node face
            continue

        # gather node coordinates
        corner_coords = coords_all[corner_nodes]  # shape (4,3)

        # 3) compute area & normal
        face_area, face_normal = compute_area_normal_rect(corner_coords)
        if face_area < 1e-14:
            continue

        # 4) face load = p * area * normal
        face_load = pressure_value * face_area * face_normal
        # distribute among 4 corners
        share = face_load / 4.0

        for n in corner_nodes:
            dof_index = 3*n
            f_global[dof_index  : dof_index+3] += share

    return f_global

def visualize_force_field(mesh, f_global, scale=1.0):
    """
    Visualize a 3D vector field (f_global) at each node of a Dolfinx mesh using PyVista.
    
    :param mesh: Dolfinx mesh object
    :param f_global: a numpy array of length (#nodes*3), 
                     storing the x,y,z force components at each node (in a block).
    :param scale: a float factor to scale the arrow glyph sizes for visibility.
    """
    # 1) Get node coords and #nodes
    coords = mesh.geometry.x          # shape = (#nodes, 3)
    num_nodes = coords.shape[0]
    
    # 2) Reshape f_global from (#nodes*3,) -> (#nodes,3)
    if f_global.size != num_nodes*3:
        raise ValueError(f"Incompatible size of f_global {f_global.size} != {num_nodes}*3")
    vecs = f_global.reshape((num_nodes, 3))

    # 3) Create a PyVista point cloud from coords
    #    We'll store "vecs" as a vector data array for each point
    #    in PyVista parlance, that's usually done by creating a PolyData
    point_cloud = pv.PolyData(coords)
    
    # Add the vector field
    point_cloud["fvec"] = vecs

    # 4) Generate glyphs (arrows) to show the vectors
    # "orient='fvec'" tells pyvista to orient each glyph along that vector
    # "scale='fvec'" can also be used, but we typically want a separate scale factor
    # We'll do scale=False to not scale glyph by vector magnitude automatically
    glyphs = point_cloud.glyph(orient="fvec", scale=False, factor=scale)
    # If you want the arrow size to reflect vector magnitude, do e.g. scale='fvec'
    # and remove factor=scale, or combine them as needed.

    # 5) PyVista plotting
    plotter = pv.Plotter()
    # Add the original points if you want them as spheres
    plotter.add_mesh(
        point_cloud,
        render_points_as_spheres=True,
        point_size=5,
        color='white',
        label='Nodes'
    )
    # Add the glyph arrows
    plotter.add_mesh(
        glyphs,
        color='red',
        label='Force Vectors'
    )

    plotter.add_legend()
    plotter.show()

def visualize_magnitude_heatmap(mesh, f_global, cmap="jet", point_size=8.0):
    """
    Visualize a discrete heatmap of vector magnitudes at each node in the mesh.
    
    Parameters:
    -----------
    mesh      : dolfinx.Mesh (with geometry.x storing node coords in shape (#nodes,3))
    f_global  : a NumPy array of length (#nodes*3), storing the x,y,z vector for each node
    cmap      : colormap (str) to use for PyVista (e.g. "jet", "coolwarm", etc.)
    point_size: size of the rendered spheres/points in the PyVista plot

    Approach:
      1. Reshape f_global -> (#nodes,3)
      2. Compute magnitude for each node
      3. Create a PyVista point cloud, store the magnitude as a scalar array
      4. Plot with that magnitude as color
    """

    coords = mesh.geometry.x
    num_nodes = coords.shape[0]
    if f_global.size != num_nodes*3:
        raise ValueError(f"f_global size {f_global.size} does not match #nodes={num_nodes}*3")

    # Reshape
    vecs = f_global.reshape((num_nodes, 3))

    # Magnitude at each node
    mags = np.linalg.norm(vecs, axis=1)

    # Create PyVista point cloud
    cloud = pv.PolyData(coords)
    # Add magnitude as a scalar array
    cloud["magnitudes"] = mags

    # Setup PyVista plotting
    plotter = pv.Plotter()
    plotter.add_mesh(
        cloud,
        scalars="magnitudes",
        render_points_as_spheres=True,
        point_size=point_size,
        cmap=cmap,
        clim=[mags.min(), mags.max()],
    )
    plotter.add_scalar_bar(
        title="Vector Magnitude",
        n_labels=5
    )
    plotter.show()

def main():
    comm = MPI.COMM_WORLD

    # 1) read mesh from XDMF
    with XDMFFile(comm, "extruded_block.xdmf", "r") as xdmf:
        # might have <Grid Name="Grid_0">
        mesh = xdmf.read_mesh(name="Grid")
        
    # 2) Find the top z
    all_z = mesh.geometry.x[:, 2]  # slice out the z-coordinates of all nodes
    z_max = all_z.max()            # the maximum z among all nodes
    print(f"Detected top z-coordinate: {z_max}")

    # 2) build the load vector
    f_vector = build_load_vector(mesh, pressure_value=5e3, z_top=z_max)

    if comm.rank == 0:
        print("Global force vector shape:", f_vector.shape)
        # maybe print a small portion
        print("First 12 entries of f_vector:", f_vector[:12])
        
    visualize_magnitude_heatmap(mesh, f_vector)

if __name__=="__main__":
    main()
