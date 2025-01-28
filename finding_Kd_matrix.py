import numpy as np
from mpi4py import MPI
import ufl

import basix
import dolfinx
from dolfinx.io import XDMFFile
from dolfinx.fem import functionspace, Function, form
from dolfinx.fem import dirichletbc, locate_dofs_geometrical
from dolfinx.fem import assemble_matrix as assemble_matrix_csr
from dolfinx.fem.petsc import assemble_matrix as assemble_matrix_petsc

def bottom_marker(x):
    # x.shape = (3, N)
    # Return True if z ~ 0 for that point
    return np.isclose(x[2], 0.0, atol=1e-9)


def add_zero_bottom_bc(mesh, V_disp):
    # 1) define the BC function for zero displacement
    #    dimension = V_disp.element.value_size
    #    we'll define a vector of all zeros
    bc_val = Function(V_disp)
    bc_val.x.array[:] = 0.0

    # 2) find dofs near z=0
    bottom_dofs = locate_dofs_geometrical(V_disp, bottom_marker)

    # 3) create the BC
    bc = dirichletbc(bc_val, bottom_dofs)
    return bc


def compute_damage_dependent_stiffness_matrix(mesh, D, E, nu, bc_disp=None):
    """
    Build the damage-dependent stiffness matrix K(D) for linear elasticity in modern Dolfinx,
    using basix.ufl.element.create_element for the vector displacement space.

    Arguments:
      mesh : dolfinx.Mesh
      D    : dolfinx.fem.Function (scalar damage field, 0 <= D <= 1)
      E, nu: Young's modulus, Poisson's ratio (granite or other material)

    Returns:
      A PETSc matrix representing K(D).
    """

    # 1) Material (Lamé) parameters
    lambda_ = E*nu / ((1+nu)*(1-2*nu))
    mu_     = E/(2*(1+nu))

    # 2) Create a vector Lagrange element using basix
    dim = mesh.topology.dim
    disp_elt = basix.ufl.element("Lagrange", basix.CellType.hexahedron, degree=1, shape=(dim,))
    # Build the function space
    V_disp = functionspace(mesh, disp_elt)

    # 3) Trial/Test
    u = ufl.TrialFunction(V_disp)
    v = ufl.TestFunction(V_disp)

    def eps(w):
        return 0.5*(ufl.grad(w) + ufl.grad(w).T)

    I = ufl.Identity(dim)
    def sigma_undamaged(u_):
        return lambda_*ufl.tr(eps(u_))*I + 2*mu_*eps(u_)

    # The PDE integrand: (1 - D)*σ(u) : ε(v)
    a_expr = (1.0 - D)*ufl.inner(sigma_undamaged(u), eps(v))*ufl.dx
    a_form = form(a_expr)

    # 4) Assemble
    A_csr = assemble_matrix_csr(a_form, bcs=[bc_disp] if bc_disp else [])  # K(D) matrix
    A_petsc = assemble_matrix_petsc(a_form, bcs=[bc_disp] if bc_disp else [])
    return A_csr, A_petsc

def main():
    comm = MPI.COMM_WORLD

    # Read a mesh from XDMF
    with XDMFFile(comm, "extruded_block.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")  # adapt if your file uses a different grid name

    # Create a scalar CG element for the damage field
    scalar_elt = basix.ufl.element("Lagrange", basix.CellType.hexahedron, degree=1)
    V_d = functionspace(mesh, scalar_elt)
    D = Function(V_d)    

    # Initialize D=0
    D.x.array[:] = 0.0
    D.x.scatter_forward()

    # Granite material
    E = 50e9
    nu = 0.25
    
    # Create a vector Lagrange element using basix    
    dim = mesh.topology.dim
    disp_elt = basix.ufl.element("Lagrange", basix.CellType.hexahedron, degree=1, shape=(dim,))
    # Build the function space
    V_disp = functionspace(mesh, disp_elt)
    
    bc_disp = add_zero_bottom_bc(mesh, V_disp)

    # Build K(D)
    A_csr, A_petsc = compute_damage_dependent_stiffness_matrix(mesh, D, E, nu, bc_disp=bc_disp)

    if comm.rank == 0:
        print("K(D) matrix shape:", A_petsc.getSize())
        
    M = A_csr.to_dense()
    
    # np.set_printoptions(precision=3, suppress=True, linewidth=900)    
    print(M)
    
    M[np.abs(M) < 0.001] = 0
    
    np.savetxt("Kd_example.csv", M, delimiter=",")
    
    diff = M - M.T
    # print(diff)
    # print(np.linalg.norm(diff))
    

if __name__ == "__main__":
    main()
