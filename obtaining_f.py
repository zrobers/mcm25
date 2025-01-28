import numpy as np
from mpi4py import MPI
import ufl
import dolfinx

from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities
from dolfinx.fem import locate_dofs_geometrical, dirichletbc, Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx.fem import functionspace
from basix.ufl import element
import basix

def main():
    comm = MPI.COMM_WORLD

    # === (1) Read the mesh from XDMF
    with XDMFFile(comm, "extruded_block.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        
    mesh.topology.create_entities(2)

    # Suppose we want top boundary => subdomain ID=1
    # We'll define a "facet_tag" that marks top boundary with ID=1, bottom with ID=2, etc.
    # For demonstration, let's do the tagging if not in the file:
    N = mesh.topology.index_map(2).size_local
    entities = np.arange(N, dtype=np.int32)
    values = np.full(N, -1, dtype=np.int32)
    
    # We'll define a small helper to locate top facets (z ~ z_max) and set them to 1
    zvals = mesh.geometry.x[:,2]
    z_max = np.max(zvals)
    # Suppose tolerance=1e-9 or so
    top_tol = 1e-9

    # locate which facets are top
    top_facets = locate_entities(mesh, 2, lambda x: np.isclose(x[2], z_max, atol=top_tol))

    # Tag them as 1
    indices = np.array(top_facets, dtype=np.int32)
    values[indices] = 1
    
    facet_tags = dolfinx.mesh.meshtags(mesh, 2, entities, values)

    # create a measure for the boundary
    # subdomain_data=facet_tags => so we can do ds(1) for top
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    # === (2) Create the displacement space (vector CG)
    dim = mesh.topology.dim
    disp_elt = element("Lagrange", basix.CellType.hexahedron, degree=1, shape=(dim,))
    V_disp = functionspace(mesh, disp_elt)

    # === (3) Define BC: zero displacement on bottom face (z=0)
    # 3.1 locate dofs
    def bottom_marker(x):
        return np.isclose(x[2], 0.0, atol=1e-9)
    from dolfinx.fem import locate_dofs_geometrical
    bottom_dofs = locate_dofs_geometrical(V_disp, bottom_marker)

    # 3.2 BC function (all zeros)
    bc_val = Function(V_disp)
    bc_val.x.array[:] = 0.0

    # 3.3 Build the BC object
    from dolfinx.fem import dirichletbc
    bc_bottom = dirichletbc(bc_val, bottom_dofs)

    # We can pass this BC to both matrix + vector assembly

    # === (4) Build K(D) if needed:
    # say we define a scalar field D=0 everywhere for now
    scalar_elt = element("Lagrange", basix.CellType.hexahedron, degree=1)
    V_d = functionspace(mesh, scalar_elt)
    D = Function(V_d)
    D.x.array[:] = 0.0

    # (Pretend we do an elasticity PDE with (1 - D)*... etc.)
    # We'll just show the matrix assembly
    # Typically you'd define a_form for your PDE
    # For brevity, let's skip the full a_form. We'll do a placeholder:
    u = ufl.TrialFunction(V_disp)
    v = ufl.TestFunction(V_disp)
    E, nu = 50e9, 0.25
    lam = E*nu/((1+nu)*(1-2*nu))
    mu  = E/(2*(1+nu))

    def eps(w):
        return 0.5*(ufl.grad(w) + ufl.grad(w).T)
    I = ufl.Identity(dim)
    def sigma_undamaged(w):
        return lam*ufl.tr(eps(w))*I + 2*mu*eps(w)

    a_expr = (1.0 - D)*ufl.inner(sigma_undamaged(u), eps(v))*ufl.dx
    a_form = form(a_expr)
    A = assemble_matrix(a_form, bcs=[bc_bottom])  # incorporate BC
    # no .assemble() if it yields a MatrixCSR
    # If you get a PETSc Mat, do A.assemble()

    # === (5) PDE-friendly assembly of footstep load vector
    # Let's define a traction on top boundary subdomain (ID=1) with magnitude = pressure_value
    pressure_value=5e3
    # uniform vertical traction downward
    traction = ufl.as_vector((0, 0, -pressure_value))

    L_expr = ufl.dot(traction, v)*ds(1)  # only integrate over facet_tag=1 (the top)
    L_form = form(L_expr)

    b = assemble_vector(L_form)
    # Apply BC => pinned bottom dofs => zero row in the final system
    # approach: apply_lifting(b, [a_form], bcs=[[bc_bottom]])
    # set_bc(b, [bc_bottom])
    # or pass bcs=[bc_bottom] directly if your version supports it
    apply_lifting(b, [a_form], bcs=[[bc_bottom]])
    from dolfinx.fem import set_bc
    set_bc(b, [bc_bottom])

    # Now b is a PETSc vector consistent with zero displacement on the bottom

    # if you want a raw NumPy array, do:
    # b_np = b.array
    # print / debug

    if comm.rank==0:
        print("Assembled K(D) and footstep vector with BCs. #dofs = ", V_disp.dofmap.index_map.size_global)
        # final shape
        # if A is a MatrixCSR, maybe do A_csr=A.to_petsc(), A_csr.assemble() ?

if __name__=="__main__":
    main()
