import numpy as np
import pyvista as pv

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


# --------------------- EXAMPLE USAGE ---------------------
if __name__ == "__main__":
    from mpi4py import MPI
    from dolfinx.io import XDMFFile
    import dolfinx

    comm = MPI.COMM_WORLD

    # Suppose we load a 3D mesh with #nodes in geometry.x
    with XDMFFile(comm, "extruded_block.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")  # or whichever <Grid Name="...">

    # Build a dummy vector field at each node (just for demonstration)
    num_nodes = mesh.geometry.x.shape[0]
    f_global = np.zeros(num_nodes*3, dtype=np.float64)

    # e.g., let's create a radial pattern from the domain center
    coords = mesh.geometry.x
    center = coords.mean(axis=0)
    for i in range(num_nodes):
        dx = coords[i] - center
        # some random amplitude
        f_global[3*i : 3*i+3] = 0.01 * dx  # small radial outward vector

    # On rank 0, let's visualize (PyVista doesn't parallelize well)
    if comm.rank == 0:
        visualize_magnitude_heatmap(mesh, f_global, cmap="viridis", point_size=10.0)
