import gmsh
from dolfinx.io import gmshio
from mpi4py import MPI

gmsh.initialize()
gmsh.model.add("TestVolume")

# Create a box with raw gmsh OCC calls
box = gmsh.model.occ.add_box(0, 0, 0, 1, 0.3, 0.1)
gmsh.model.occ.synchronize()

# Mesh in 3D
gmsh.model.mesh.generate(3)

# Convert
mesh, ct, pt = gmshio.model_to_mesh(
    gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=3
)

gmsh.finalize()

if mesh.comm.rank == 0:
    print("Mesh loaded into Dolfinx. #cells:", mesh.topology.index_map(mesh.topology.dim).size_local)