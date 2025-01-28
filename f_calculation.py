import numpy as np
from mpi4py import MPI
import pyvista as pv

# Dolfinx
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary

def top_marker(x, z_top=0.1, tol=1e-9):
    """
    Mark top boundary facets if their corner nodes
    satisfy z ~ z_top (within tol). x.shape = (3,N).
    """
    return np.isclose(x[2], z_top, atol=tol)

def compute_area_normal_rect(coords: np.ndarray):
    """
    Compute area & normal for a rectangular face (4 corners).
    coords shape=(4,3). 
    We do cross( (1)-(0), (3)-(0) ).
    If your face is not exactly planar or you have more corners,
    adapt or do 'fan triangulation'.
    """
    if coords.shape != (4,3):
        raise ValueError("This function needs exactly 4 corner nodes.")
    v1 = coords[1] - coords[0]
    v2 = coords[3] - coords[0]
    cross_vec = np.cross(v1, v2)
    area = np.linalg.norm(cross_vec)
    if area < 1e-14:
        return 0.0, np.zeros(3)
    normal = cross_vec / area
    return area, normal

def build_load_vector(
    mesh,
    pressure_value=1e4,
    z_top=0.1,
    # Ellipse parameters:
    ellipse_center=(0.5,0.15),  # (cx,cy)
    a=0.2,                      # major radius in x
    b=0.07                      # minor radius in y
):
    """
    1) locate top boundary facets at z~z_top
    2) for each face with 4 corners, compute midpoint,
       check if in ellipse => if yes, compute area & normal
    3) distribute face load among corner nodes
    return a (#nodes*3,) array for f_global
    """

    coords_all = mesh.geometry.x
    num_nodes = coords_all.shape[0]
    f_global = np.zeros(num_nodes*3, dtype=np.float64)

    # locate top facets
    top_facets = locate_entities_boundary(
        mesh, dim=2,
        marker=lambda x: top_marker(x, z_top=z_top)
    )

    # ensure face->node connectivity
    c = mesh.topology.connectivity(2,0)
    if c is None:
        mesh.topology.create_connectivity(2,0)
        c = mesh.topology.connectivity(2,0)

    offsets = c.offsets
    node_array = c.array

    # We'll define a helper to get the midpoint of a face
    # assuming 4 corners:
    def face_midpoint(facet_nodes):
        ccoords = coords_all[facet_nodes]
        return ccoords.mean(axis=0)  # shape(3,)

    for fidx in top_facets:
        start = offsets[fidx]
        end   = offsets[fidx+1]
        corner_nodes = node_array[start:end]
        if len(corner_nodes) != 4:
            continue  # skip if not 4-corner face

        # 1) check midpoint in ellipse
        midpoint = face_midpoint(corner_nodes)    # (x_m,y_m,z_m)
        dx = (midpoint[0] - ellipse_center[0])/a
        dy = (midpoint[1] - ellipse_center[1])/b
        if dx*dx + dy*dy > 1.0:
            # outside ellipse => skip load
            continue

        # 2) compute area & normal
        corner_coords = coords_all[corner_nodes]
        face_area, face_normal = compute_area_normal_rect(corner_coords)
        if face_area < 1e-14:
            continue

        # 3) face load = p * area * normal
        face_load = pressure_value * face_area * face_normal
        share = face_load / 4.0

        for n in corner_nodes:
            dof_index = 3*n
            f_global[dof_index:dof_index+3] += share

    return f_global

# -------------- Visualization code --------------
def visualize_force_field(mesh, f_global, scale=1.0):
    coords = mesh.geometry.x
    num_nodes = coords.shape[0]
    if f_global.size != num_nodes*3:
        raise ValueError("f_global size mismatch.")
    vecs = f_global.reshape((num_nodes,3))

    point_cloud = pv.PolyData(coords)
    point_cloud["fvec"] = vecs

    glyphs = point_cloud.glyph(orient="fvec", scale=False, factor=scale)

    plotter = pv.Plotter()
    plotter.add_mesh(
        point_cloud,
        render_points_as_spheres=True,
        point_size=5,
        color='white',
        label='Nodes'
    )
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
    from dolfinx.io import XDMFFile
    comm = MPI.COMM_WORLD

    # read mesh from XDMF
    with XDMFFile(comm, "extruded_block.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")  # adjust grid name if needed

    # find top z
    z_vals = mesh.geometry.x[:,2]
    z_top = z_vals.max()
    print(f"Detected top z = {z_top}")

    # define ellipse center, axes
    ellipse_center = (0.5, 0.15)
    a, b = 0.6, 0.3
    # build f_global
    f_global = build_load_vector(
        mesh,
        pressure_value=5e4,
        z_top=z_top,
        ellipse_center=ellipse_center,
        a=a,
        b=b
    )
    
    if comm.rank == 0:
        print("f_global shape:", f_global.shape)
        # visualize
        visualize_magnitude_heatmap(mesh, f_global)

if __name__=="__main__":
    main()
