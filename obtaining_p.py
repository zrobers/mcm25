import numpy as np
from mpi4py import MPI
import ufl
import basix
import dolfinx

from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities
from dolfinx.fem import (functionspace, Function, form, dirichletbc, locate_dofs_geometrical, 
                         set_bc, apply_lifting)
from dolfinx.fem import assemble_matrix as assemble_matrix_csr
from dolfinx.fem.petsc import assemble_matrix as assemble_matrix_petsc
from petsc4py import PETSc
from basix.ufl import element


##############################################################################
# PART 1: K(D) code
##############################################################################

#eps and sigma calculation
def eps(w):
    return 0.5*(ufl.grad(w) + ufl.grad(w).T)

def sigma_undamaged(u_, dim, lambda_, mu_):
    return lambda_*ufl.tr(eps(u_))*ufl.Identity(dim) + 2*mu_*eps(u_)


def bottom_marker(x):
    """
    x.shape = (3, N). Return True if z ~ 0 for that point => pinned bottom face.
    """
    return np.isclose(x[2], 0.0, atol=1e-9)

def add_zero_bottom_bc(mesh, V_disp):
    """
    Creates a Dirichlet BC for zero displacement on the bottom face (z=0).
    """
    bc_val = Function(V_disp)
    bc_val.x.array[:] = 0.0
    bottom_dofs = locate_dofs_geometrical(V_disp, bottom_marker)
    bc = dirichletbc(bc_val, bottom_dofs)
    return bc

def compute_damage_dependent_stiffness_matrix(mesh, D, E, nu, bc_disp=None):
    """
    Build the damage-dependent stiffness matrix K(D) for linear elasticity in Dolfinx

    Returns (A_csr, A_petsc):
      A_csr   = a Dolfinx 'MatrixCSR' object (if you want to .to_dense()),
      A_petsc = a PETSc Mat for final solves.
    """
    # Lame parameters
    lambda_ = E*nu / ((1+nu)*(1-2*nu))
    mu_     = E/(2*(1+nu))

    # Vector FE space for displacement
    dim = mesh.topology.dim
    disp_elt = basix.ufl.element("Lagrange", basix.CellType.hexahedron, degree=1, shape=(dim,))
    V_disp = functionspace(mesh, disp_elt)

    # Bilinear form: a(u,v) = ∫ (1-D)*sigma_undamaged(u):eps(v) dx
    u = ufl.TrialFunction(V_disp)
    v = ufl.TestFunction(V_disp)

    a_expr = (1.0 - D)*ufl.inner(sigma_undamaged(u, dim, lambda_, mu_), eps(v))*ufl.dx
    a_form = form(a_expr)

    # Assemble both a MatrixCSR and a PETSc Mat
    A_csr   = assemble_matrix_csr(a_form, bcs=[bc_disp] if bc_disp else [])
    A_petsc = assemble_matrix_petsc(a_form, bcs=[bc_disp] if bc_disp else [])
    return A_csr, A_petsc

##############################################################################
def compute_pressure_top(
    mesh,
    u_sol,          # displacement solution in volume
    lambda_, mu_,   # Lame constants
    top_facets,     # np.array of local facet indices for the top boundary
    facet_tags,     # MeshTags storing ID=1 for top boundary
    top_id=1
):
    """
    Boundary-based approach to find p = n^T sigma(u_sol) n 
    on the facets with ID=top_id (the top face).
    
    We define a 'DG(0)' or 'DG(1)' space in the domain 
    that can store face-based dofs, then 'project' or 
    'interpolate' the boundary expression onto that space, 
    zeroing out dofs not on top facets. 
    """

    # (A) Setup for boundary measure ds with subdomain_data=facet_tags
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    # (B) Define the normal via FacetNormal
    n = ufl.FacetNormal(mesh)

    # define eps, sigma
    def eps(w):
        return 0.5*(ufl.grad(w) + ufl.grad(w).T)
    I = ufl.Identity(mesh.topology.dim)
    def sigma_(w):
        return lambda_*ufl.tr(eps(w))*I + 2.0*mu_*eps(w)

    # contact pressure on top boundary => p_expr = n^T sigma(u_sol) n
    stress_expr = sigma_(u_sol)
    p_expr = ufl.dot(n, stress_expr)
    p_expr = ufl.dot(p_expr, n)  # = (n^T sigma(u) ) dot n

    # (C) Build a 'DG' space that has a dof per facet, e.g. DG(0).
    # In Dolfinx, a typical approach is:
    #   V_bd_elt = element("Discontinuous Lagrange", celltype, degree=0) 
    # that yields a constant dof per cell; we can use 'cell' or 'facet'. 
    # We'll do a cell-based DG, then restrict to facets. 
    # For a simpler approach, we can store 1 dof/cell, then 
    # we do an integral over the boundary. We'll have to isolate the boundary dofs though.

    # Let's do DG(0) in the domain:
    dg_elt = element("Discontinuous Lagrange", basix.CellType.hexahedron, degree=0)
    V_bd = dolfinx.fem.functionspace(mesh, dg_elt)

    # (D) Build a 'projection' form that integrates p_expr over boundary facets 
    # and uses test function in V_bd. We'll do:
    #   a(q, phi) = ∫ q * phi dx  (BUT we only want boundary?)
    # Actually, to store boundary data in a cell-based DG approach, we do a partial approach:
    #
    # We'll do an integral over 'ds( top_id )' with p_expr * phi. 
    # Then if we want to solve for q s.t. q ~ p_expr on the boundary, we'd do:
    #   a(q, phi) = ∫ q * phi ds,    L( phi ) = ∫ p_expr * phi ds
    # That yields dofs in each cell that touches the top boundary. 
    #
    # It's a bit approximate: cell-based DG(0) lumps the boundary value into the entire cell. 
    # For a single top layer of cells, it's workable.

    q_trial = ufl.TrialFunction(V_bd)
    phi     = ufl.TestFunction(V_bd)

    a_bd_expr = q_trial * phi * ds(top_id)
    L_bd_expr = p_expr * phi * ds(top_id)

    a_bd = dolfinx.fem.form(a_bd_expr)
    L_bd = dolfinx.fem.form(L_bd_expr)

    # (E) assemble system
    A_bd = assemble_matrix_petsc(a_bd)
    A_bd.assemble()

    b_bd = dolfinx.fem.petsc.assemble_vector(L_bd)
    # no BCs to apply for a boundary "projection" if we interpret it as a mass matrix approach
    # we can do a minimal approach => solve A_bd q = b_bd

    # (F) Solve for q
    q_vec = create_vector(A_bd) #####
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A_bd)
    ksp.setFromOptions()
    ksp.solve(b_bd, q_vec)

    # (G) store in a function in V_bd
    p_func = dolfinx.fem.Function(V_bd)
    p_func.x.setArray(q_vec.array)
    p_func.x.scatter_forward()

    return p_funcs

def apply_archard_wear_from_arrays(
    mesh: dolfinx.mesh.Mesh,
    top_nodes: np.ndarray,
    foot_load_data: np.ndarray,
    friction_slip_data: np.ndarray,
    archard_k: float=1.0
) -> None:
    """
    Applies Archard's law node-by-node on the top boundary of the stair mesh.
    The formula is: delta_h = archard_k * pressure * slip
    We subtract delta_h from the z-coordinate of each top node.

    Arguments:
    -----------
    mesh : The Dolfinx Mesh to be updated
    top_nodes : 1D array of node indices that belong to the top layer
    foot_load_data : shape (#top_nodes,)
                    The scalar foot load (pressure) at each top node
    friction_slip_data : shape (#top_nodes,)
                         The scalar friction slip at each top node
    archard_k : archard constant (default=1.0)
    
    Returns:
    -----------
    None (mesh.geometry.x is updated in-place)
    """

    comm = mesh.comm

    # Quick check
    if len(top_nodes) != len(foot_load_data) or len(top_nodes) != len(friction_slip_data):
        raise ValueError("Mismatch in lengths of top_nodes, foot_load_data, friction_slip_data")

    coords = mesh.geometry.x  # shape (#mesh_nodes, 3)

    # For demonstration, we do a trivial "pre-processing" step on frictional slip
    # e.g., if user gave slip in mm, convert to meters, or do scaling
    friction_slip_data_m = friction_slip_data * 1.0e-3  # example: from mm -> m

    # Now apply Archard node by node
    for i, node_index in enumerate(top_nodes):
        p = foot_load_data[i]
        s = friction_slip_data_m[i]  # "processed" slip
        delta_h = archard_k * p * s
        
        if abs(delta_h) < 1e-14:
            continue
        
        coords[node_index, 2] -= delta_h  # lower the z-coord by delta_h

    # finalize in parallel
    mesh.geometry.x[...] = coords
    comm.barrier()

# ------------------------------------------------------------------------
# Minimal usage example
def demo_archard_application():
    """
    Illustrates how to use apply_archard_wear_from_arrays 
    with a small example mesh and dummy data.
    """
    from dolfinx.io import XDMFFile

    comm = MPI.COMM_WORLD
    # 1) Load some small stair mesh
    with XDMFFile(comm, "stair_mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    # 2) Identify top-layer nodes. Suppose we define a function
    #    that returns all nodes with z=some top layer...
    coords = mesh.geometry.x
    z_vals = coords[:,2]
    z_max = np.max(z_vals)
    top_tol = 1e-9
    top_nodes = np.where(np.isclose(z_vals, z_max, atol=top_tol))[0]

    # 3) Suppose user gave us foot_load_data and friction_slip_data 
    #    as arrays of length= len(top_nodes)
    #    We'll create dummy data
    foot_load_data = np.random.uniform(1.0, 2.0, size=len(top_nodes))   # e.g. 1-2 kPa
    friction_slip_data = np.random.uniform(0.1, 0.2, size=len(top_nodes))  # e.g. 0.1-0.2 mm ?

    # 4) Apply Archard
    archard_k = 1.0  # user chosen
    apply_archard_wear_from_arrays(
        mesh, top_nodes,
        foot_load_data, friction_slip_data,
        archard_k=archard_k
    )

    if comm.rank == 0:
        print("Updated mesh coords after Archard application (sample):\n", mesh.geometry.x[:10])


##############################################################################
# PART 5: loop
##############################################################################

def main():
    
    #pre-loop processing
    comm = MPI.COMM_WORLD

    # (A) Read the mesh from XDMF
    with XDMFFile(comm, "extruded_block.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    # We ensure 2D entities exist so we can locate facets
    mesh.topology.create_entities(2)

    # (B) Mark the top boundary with subdomain ID=1
    N = mesh.topology.index_map(2).size_local
    entities = np.arange(N, dtype=np.int32)
    values   = np.full(N, -1, dtype=np.int32)

    # find top facets (z ~ z_max)
    zvals = mesh.geometry.x[:,2]
    z_max = np.max(zvals)
    top_tol = 1e-9
    top_facets = locate_entities(mesh, 2, lambda x: np.isclose(x[2], z_max, atol=top_tol))
    indices = np.array(top_facets, dtype=np.int32)
    values[indices] = 1

    facet_tags = dolfinx.mesh.meshtags(mesh, 2, entities, values)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    # (C) Create displacement space, define BC for bottom face
    dim = mesh.topology.dim
    disp_elt = element("Lagrange", basix.CellType.hexahedron, degree=1, shape=(dim,))
    V_disp   = functionspace(mesh, disp_elt)

    def bottom_marker(x):
        return np.isclose(x[2], 0.0, atol=1e-9)

    bottom_dofs = locate_dofs_geometrical(V_disp, bottom_marker)
    bc_val = Function(V_disp)
    bc_val.x.array[:] = 0.0
    bc_bottom = dirichletbc(bc_val, bottom_dofs)

    # (D) Create a scalar space for damage, set D=0
    scalar_elt = element("Lagrange", basix.CellType.hexahedron, degree=1)
    V_d = functionspace(mesh, scalar_elt)
    D = Function(V_d)
    D.x.array[:] = 0.0
    D.x.scatter_forward()

    # (E) Build the 'K(D)' matrix from your code
    E=50e9
    nu=0.25
    A_csr, A_petsc = compute_damage_dependent_stiffness_matrix(mesh, D, E, nu, bc_disp=bc_bottom)
    # for PDE solves, we typically use the PETSc Mat
    A_petsc.assemble()

    # (F) Build the footstep load vector in PDE-friendly style
    # "pressure_value=5e3" means a uniform vertical traction on top

    # define the PDE integrand: traction = (0,0,-pressure_value)
    # L(v) = ∫ traction·v ds(1)
    pressure_value = 5e3
    traction = ufl.as_vector((0,0,-pressure_value))

    print(traction)

    u = ufl.TrialFunction(V_disp)
    v = ufl.TestFunction(V_disp)

    L_expr = ufl.dot(traction, v)*ds(1)
    L_form = form(L_expr)
    
    b_petsc = dolfinx.fem.petsc.assemble_vector(L_form)
    
    # Apply BC to b
    # We define or reuse the same 'a_form' used in K(D) => let's do minimal approach
    # We'll define a dummy a_form in the same space:
    def eps(w):
        return 0.5*(ufl.grad(w) + ufl.grad(w).T)
    lam = E*nu/((1+nu)*(1-2*nu))
    mu  = E/(2*(1+nu))
    I = ufl.Identity(dim)
    def sigma_undamaged(w):
        return lam*ufl.tr(eps(w))*I + 2*mu*eps(w)
    a_expr_dummy = (1.0 - D)*ufl.inner(sigma_undamaged(u), eps(v))*ufl.dx
    a_form_dummy = form(a_expr_dummy)

    A_dummy = assemble_matrix_petsc(a_form_dummy, bcs=[bc_bottom])
    A_dummy.assemble()

    # Apply BC to b
    
    
    apply_lifting(b_petsc, [a_form_dummy], bcs=[[bc_bottom]])
    set_bc(b_petsc, [bc_bottom])

    # Now b_petsc is the final PDE vector (PETSc Vec form).
    # We'll do a PDE solve: A_petsc u = b_petsc

    # (G) Solve PDE
    # create solution vector
    x_sol = dolfinx.fem.petsc.create_vector(a_form_dummy)
    # we have b_petsc in a PETSc Vec => must finalize 
    b_petsc.assemble()

    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setOperators(A_petsc)
    ksp.setFromOptions()
    ksp.solve(b_petsc, x_sol)

    # place solution in a Function
    u_sol = Function(V_disp)
    u_sol.x.array[:] = x_sol.array
    u_sol.x.scatter_forward()

    if comm.rank==0:
        print("System solve done. #dofs:", len(x_sol.array))

    # (H) Optionally compute contact pressure 'p' on top for Archard:
    # We'll do a placeholder function here:

    p_map = compute_pressure_top(mesh, u_sol, D, E, nu, facet_tags, top_id=1)
    if comm.rank == 0:
        print("p_map on top boundary (placeholder):", p_map)




if __name__ == "__main__":
    main()
