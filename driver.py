import numpy as np
from mpi4py import MPI
import ufl

import basix
import dolfinx
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities
from dolfinx.fem import (functionspace, Function, form, dirichletbc, locate_dofs_geometrical,
                         assemble_vector, apply_lifting, set_bc)
from dolfinx.fem.petsc import assemble_matrix as assemble_matrix_petsc
from dolfinx.fem.petsc import assemble_vector
from dolfinx.la import create_petsc_vector
from dolfinx.mesh import meshtags
from petsc4py import PETSc
from basix.ufl import element

##############################################################################
# Helpers
##############################################################################

def lame_constants(E, nu):
    lambda_ = E*nu/((1+nu)*(1-2*nu))
    mu_     = E/(2*(1+nu))
    return lambda_, mu_

def bottom_marker(x):
    return np.isclose(x[2], 0.0, atol=1e-9)

def create_displacement_space(mesh):
    dim = mesh.topology.dim
    disp_elt = element("Lagrange", basix.CellType.hexahedron, degree=1, shape=(dim,))
    return functionspace(mesh, disp_elt)

def add_zero_bottom_bc(mesh, V_disp):
    bottom_dofs = locate_dofs_geometrical(V_disp, bottom_marker)
    bc_val = Function(V_disp)
    bc_val.x.array[:] = 0.0
    bc_bottom = dirichletbc(bc_val, bottom_dofs)
    return bc_bottom

def create_top_facet_tags(mesh, z_tolerance=1e-9, top_node_value=1):
    """
    Create a MeshTags object labeling the top boundary facets with ID=1
    and setting all other facets to -1.
    Return (facet_tags, top_id).
    """

    # 1) The dimension for facets in a 3D mesh is 2
    facet_dim = 2

    # Ensure we have connectivity for dimension 2
    mesh.topology.create_entities(facet_dim)

    # The total # of local facets
    num_facets_local = mesh.topology.index_map(facet_dim).size_local

    # We'll create an array of facet indices: [0,1,...,num_facets_local-1]
    entities = np.arange(num_facets_local, dtype=np.int32)
    # And the corresponding marker array, default = -1
    values = np.full(num_facets_local, -1, dtype=np.int32)

    # 2) We  a function that identifies the top boundary facets:
    def top_facet_marker(x):
        # x.shape = (3, num_points)
        # we say "true" if z ~ z_max
        z_vals = x[2]
        z_max = np.max(mesh.geometry.x[:,2])
        return np.isclose(z_vals, z_max, atol=z_tolerance)

    # 3) locate the top facets by dimension=2
    top_facets = locate_entities(mesh, facet_dim, top_facet_marker)
    # Then we label them with ID=1
    values[top_facets] = top_node_value

    # 4) Create the mesh tags
    facet_tags = meshtags(mesh, dim=facet_dim, entities=entities, values=values)
    return facet_tags

def eps(u):
    return 0.5*(ufl.grad(u)+ufl.grad(u).T)

##############################################################################
# 1) Build PDE with boundary traction from node-based foot load
##############################################################################

def create_dg0_foot_load(mesh, V_disp, top_nodes, foot_load_per_node, facet_tags=None, top_id=1):
    """
    Create a DG(0) function in the domain from node-based top boundary loads.
    We'll average the top node data among each cell that touches the top boundary,
    storing the result in a DG(0) function "foot_func".
    """
    
    dg_0_elt = element("Discontinuous Lagrange", basix.CellType.hexahedron, degree=0)
    V_dg = dolfinx.fem.functionspace(mesh, dg_0_elt)  # scalar DG(0)

    c_to_n = mesh.topology.connectivity(mesh.topology.dim, 0)
    if c_to_n is None:
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        c_to_n = mesh.topology.connectivity(mesh.topology.dim, 0)
    offsets = c_to_n.offsets
    node_array = c_to_n.array

    c_start, c_end = mesh.topology.index_map(mesh.topology.dim).local_range

    # 1) Create a DG(0) function in the domain
    foot_func = dolfinx.fem.Function(V_dg)
    dg_data = foot_func.x.array  # The dof array for the function (not the space)
    dg_data[:] = 0.0             # Initialize to zero

    # 2) For each local cell, find how many top nodes it has, average those loads
    for c in range(c_start, c_end):
        start = offsets[c]
        end   = offsets[c+1]
        c_nodes = node_array[start:end]

        sum_ = 0.0
        count_ = 0
        for nd in c_nodes:
            idx_list = np.where(top_nodes == nd)[0]
            if len(idx_list) > 0:
                i_nd = idx_list[0]
                sum_ += foot_load_per_node[i_nd]
                count_ += 1

        val = sum_ / count_ if count_ > 0 else 0.0

        # cell_dofs(...) is a method
        c_dofs = V_dg.dofmap.cell_dofs(c)
        dof_idx = c_dofs[0]  # DG(0) => 1 dof per cell
        dg_data[dof_idx] = val

    # 3) finalize in parallel
    foot_func.x.scatter_forward()
    
    v = ufl.TestFunction(V_disp)
    n = ufl.FacetNormal(mesh)

    ds = ufl.Measure("ds", domain=mesh, subdomain_id = top_id, subdomain_data=facet_tags)
    traction_expr = ufl.as_vector(foot_func * n)
    L_expr = ufl.dot(traction_expr, v)*ds
    L_form = form(L_expr)

    b_petsc = assemble_vector(L_form)
        
    return b_petsc

def apply_dirichlet_bc_to_vector(b_petsc, a_form, bc_bottom):
    """
    Apply the same BC to the vector b_petsc, using 'apply_lifting' + 'set_bc'.
    After that, 'b_petsc.assemble()'.
    """
    # Convert to local array for the apply_lifting
    apply_lifting(b_petsc, [a_form], bcs=[[bc_bottom]])
    set_bc(b_petsc, [bc_bottom])
    b_petsc.assemble()
    
    return b_petsc


def build_KD_system(mesh, V_disp, bc_bottom, D, lam, mu):
    """
    Build the matrix for K(D). ignoring the load vector so far
    """
    u = ufl.TrialFunction(V_disp)
    v = ufl.TestFunction(V_disp)

    def eps(w):
        return 0.5*(ufl.grad(w)+ufl.grad(w).T)
    I=ufl.Identity(mesh.topology.dim)
    def sigma_undmg(w):
        return lam*ufl.tr(eps(w))*I + 2*mu*eps(w)

    a_expr = (1.0 - D)*ufl.inner(sigma_undmg(u), eps(v))*ufl.dx
    a_form = dolfinx.fem.form(a_expr)

    A = assemble_matrix_petsc(a_form, bcs=[bc_bottom])
    A.assemble()
    return A, a_form


##############################################################################
# compute boundary contact pressure from displacement
##############################################################################

def compute_contact_pressure_on_boundary(mesh, u_sol, facet_tags, lam, mu, top_id=1):
    """
    Boundary approach: p = n^T sigma(u) n on ds(top_id).
    We'll store p in a DG(0) function => 1 dof/cell, but only relevant for top boundary cells.
    """
    from basix.ufl import element as scalar_elt
    dg_elt = scalar_elt("Discontinuous Lagrange", basix.CellType.hexahedron, 0)
    V_dg = dolfinx.fem.functionspace(mesh, dg_elt)
    p_func = dolfinx.fem.Function(V_dg)
    p_data = p_func.x.array

    # adjacency cell->node for partial use
    c_to_n = mesh.topology.connectivity(mesh.topology.dim, 0)
    if c_to_n is None:
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        c_to_n = mesh.topology.connectivity(mesh.topology.dim, 0)
    offsets = c_to_n.offsets
    node_array = c_to_n.array

    def eps(w):
        return 0.5*(ufl.grad(w)+ufl.grad(w).T)
    I=ufl.Identity(mesh.topology.dim)
    def sigma_(w):
        return lam*ufl.tr(eps(w))*I + 2*mu*eps(w)

    # We'll do a facet normal approach => n^T sigma(u) n
    # For each cell on top boundary, we do an integral to get average or do a single sample.
    # We'll do a naive approach: sample midpoint of the cell's top facet? 
    # or do a "mass matrix approach" => let's do partial approach for demonstration.

    # For a real approach, define a form:
    #   p_expr = n^T sigma(u_sol) n on ds(top_id)
    # and project. We'll do a simplified node sampling approach for demonstration:

    # local cell range
    c_start, c_end = mesh.topology.index_map(mesh.topology.dim).local_range
    for c in range(c_start, c_end):
        # check if cell c touches top boundary => if it has a facet with facet_tags=top_id
        # We'll do a minimal approach: we skip detail. 
        # If you want a formal approach, you do a ds measure + project. 
        # We'll do a placeholder => p_data[dof] = some small value
        c_dofs = V_dg.dofmap.cell_dofs(c)
        dof_idx = c_dofs[0]
        p_data[dof_idx] = 0.0

    p_func.x.scatter_forward()
    return p_func

def establish_relative_eval_location(dx, dy, dz):
    center_top = (dx/2, dy/2, dz)
    


def gather_contact_pressure_at_nodes(mesh, p_func, top_nodes):
    """
    Sample p_func at each top node coordinate, returning
    an array contact_pressure_at_node of shape (len(top_nodes),).
    For small or serial runs, we do a naive local `eval()`. 
    In parallel, you might need a gather approach if some top nodes
    belong to different ranks or the function does not own them all.
    """
    
    coords = mesh.geometry.x
    contact_pressure_at_node = np.zeros(len(top_nodes), dtype=np.float64)

    for i, nd in enumerate(top_nodes):
        x_ = coords[nd]
        val = 0
        for j, _ in enumerate(top_nodes):
        # Evaluate p_func at x_ => returns a 1D array (since p_func is scalar, shape(1,))
            val += p_func.eval(x_, j)  # returns (array_of_values, cell_indices)
        if val != 0:
            contact_pressure_at_node[i] = val  # the first (and only) component
        else:
            # might happen if x_ is not found locally in parallel
            contact_pressure_at_node[i] = 0.0

    return contact_pressure_at_node

def apply_archard_wear_nodewise(mesh, top_nodes, contact_pressure_at_node, slip_array, archard_k=1.0):
    coords = mesh.geometry.x
    for i, nd in enumerate(top_nodes):
        dh = archard_k*contact_pressure_at_node[i]*slip_array[i]
        coords[nd,2] -= dh
    mesh.geometry.x[...] = coords
    mesh.comm.barrier()

##############################################################################
# Fatigue model
##############################################################################

def update_damage_lemaitre(
    mesh: dolfinx.mesh.Mesh,
    D: dolfinx.fem.Function,
    u_sol: dolfinx.fem.Function,
    alpha: float = 1e-4,
    m: float = 2.0,
    lam: float = 50e9,
    mu: float = 0.25
):
    """
    Lemaitre-style damage update:
      D_{n+1} = min(D_n + alpha*(sigma_eq)^m, 1.0)
    where sigma_eq is the "equivalent stress" (e.g. Von Mises) computed from the PDE solution.

    Steps:
      1) compute sigma(u_sol) in each cell (DG(0))
      2) compute eq_stress = von_mises( sigma ) or any measure
      3) read eq_stress dofs => eq_stress_array
      4) oldD = D.x.array
         newD = min( oldD + alpha*( eq_stress_array^m ), 1.0 )
      5) scatter forward
    """

    # (A) Create a DG(0) space for the stress measure
    dg_elt = element("Discontinuous Lagrange", basix.CellType.hexahedron, degree=0)
    V_dg = dolfinx.fem.functionspace(mesh, dg_elt)
    stress_func = dolfinx.fem.Function(V_dg)
    stress_data = stress_func.x.array
    stress_data[:] = 0.0  # init to 0

    # (B) define "von Mises" or "some eq. stress" from PDE solution
    #   sigma = lam*tr(eps(u))*I + 2*mu*eps(u)
    #   von_mises( sigma ) = sqrt(3/2 * s_dev : s_dev)


    # define expressions
    from ufl import grad, Identity, TrialFunction, TestFunction
    dim = mesh.topology.dim

    def eps(w):
        return 0.5*(grad(w) + grad(w).T)

    I = ufl.Identity(dim)
    def sigma_(w):
        return lam*ufl.tr(eps(w))*I + 2*mu*eps(w)

    sigma_expr = sigma_(u_sol)
    # dev(sigma) = sigma - 1/3 tr(sigma)*I for 3D
    s_dev = sigma_expr - (ufl.tr(sigma_expr)/dim)*I
    # eq_stress_expr = sqrt( (3/2) * s_dev : s_dev ) = Von Mises
    eq_stress_expr = ufl.sqrt(1.5 * ufl.inner(s_dev, s_dev))

    # (C) Project eq_stress_expr onto DG(0) => stress_func
    # We'll do a "mass matrix" approach, i.e. "trial = eq_stress_expr"
    trial = ufl.TrialFunction(V_dg)
    test  = ufl.TestFunction(V_dg)

    a_proj = dolfinx.fem.form(ufl.inner(trial, test)*ufl.dx)
    L_proj = dolfinx.fem.form(ufl.inner(eq_stress_expr, test)*ufl.dx)

    A_proj = dolfinx.fem.petsc.assemble_matrix(a_proj)
    A_proj.assemble()

    b_proj = dolfinx.fem.petsc.assemble_vector(L_proj)
    # solve A_proj * stress_func = b_proj

    x_proj = dolfinx.fem.petsc.create_vector(L_proj)
    ksp = PETSc.KSP().create(mesh.comm)
    
    ksp.setOperators(A_proj)
    ksp.setFromOptions()    
    ksp.solve(b_proj, x_proj)
    
    stress_func.x.array[:] = x_proj.array
    stress_func.x.scatter_forward()

    # (D) read eq_stress data from stress_func.x.array => eq_stress_array
    eq_stress_array = stress_func.x.array[:]
    
    print(eq_stress_array.shape)
    # (E) Update D.x.array => D_{n+1} = min(D_n + alpha*( eq_stress^m ), 1.0)
    oldD = D.x.array
    newD = oldD + alpha * np.power(eq_stress_array, m)
    newD = np.minimum(newD, 1.0)
    D.x.array[:] = newD
    D.x.scatter_forward()



##############################################################################
# time loop
##############################################################################

def main_time_loop():
    comm = MPI.COMM_WORLD
    
    # define material parameters
    E, nu= 50, 0.25
    lam, mu = lame_constants(E, nu)

    with XDMFFile(comm, "extruded_block.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        
    mesh.topology.create_entities(2)

    # build displacement space
    V_disp = create_displacement_space(mesh)
    bc_bottom = add_zero_bottom_bc(mesh, V_disp)

    # define facet_tags for top boundary
    facet_tags = create_top_facet_tags(mesh, z_tolerance=1e-9)

    # build damage function
    from basix.ufl import element as scalar_elt
    from mpl_toolkits.mplot3d import Axes3D
    Vd = dolfinx.fem.functionspace(mesh, scalar_elt("Lagrange", basix.CellType.hexahedron, 1))
    D = dolfinx.fem.Function(Vd)
    D.x.array[:] = 0.0

    # identify top nodes
    coords_ = mesh.geometry.x
    z_ = coords_[:,2]
    z_max = np.max(z_)
    top_tol=1e-9
    top_nodes = np.where(np.isclose(z_, z_max, atol=top_tol))[0]

    # if we want a dictionary for top_node -> index
    top_node_to_idx = {}
    for i, nd in enumerate(top_nodes):
        top_node_to_idx[nd] = i
    
    # time loop
    num_steps=25
    for step in range(num_steps):
        # read node-based foot load + friction slip from external data
        # e.g. foot_node_data = [ (p_i, s_i),  (p_j, s_j), ... ] for i in top_nodes
        # or a separate arrays => let's do arrays for demonstration
        foot_load_arr = np.random.uniform(0.5, 1.5, size=len(top_nodes))   # e.g. "pressure"
        friction_slip_arr = np.random.uniform(1, 2, size=len(top_nodes))

        # build PDE matrix K(D)
        A_petsc, a_form = build_KD_system(mesh, V_disp, bc_bottom, D, lam, mu)

        # we find the top facets and set them to 1 in tags
        top_id = 1
        facet_tags = create_top_facet_tags(mesh, z_tolerance=1e-9, top_node_value = top_id)

        # real boundary traction from foot_load_arr:
        b_petsc = create_dg0_foot_load(mesh, V_disp, top_nodes, foot_load_arr, facet_tags = facet_tags, top_id = top_id)
        b_petsc = apply_dirichlet_bc_to_vector(b_petsc, a_form, bc_bottom)

        # create solution vector from the bilinear form
        x_sol = dolfinx.fem.petsc.create_vector(a_form)
        
        # Solve the PDE
        ksp = PETSc.KSP().create(mesh.comm)
        ksp.setOperators(A_petsc)
        ksp.setFromOptions()
        ksp.solve(b_petsc, x_sol)

        # store solution in a displacement Function
        u_sol = Function(V_disp)
        u_sol.x.array[:] = x_sol.array
        u_sol.x.scatter_forward()

        # compute boundary contact pressure from the displacement
        # p_func = compute_contact_pressure_on_boundary(mesh, u_sol, facet_tags, lam, mu, top_id )
        contact_pressure_at_node = friction_slip_arr

        # archard => update geometry
        apply_archard_wear_nodewise(mesh, top_nodes, contact_pressure_at_node, friction_slip_arr)

        # update_damage_lemaitre(mesh, D, u_sol, alpha=1e-4, m=2.0)
        
        # Calculate displacement magnitude for each node
        displacement_magnitude = np.linalg.norm(u_sol.x.array.reshape((-1, 3)), axis=1)

        # # Create a plotly graph
        # import plotly.graph_objects as go

        # # Extract node coordinates
        # node_coords = mesh.geometry.x

        # # Create a scatter plot of the nodes with color representing displacement magnitude
        # scatter = go.Scatter3d(
        #     x=node_coords[:, 0],
        #     y=node_coords[:, 1],
        #     z=node_coords[:, 2],
        #     mode='markers',
        #     marker=dict(
        #     size=3,
        #     color=displacement_magnitude,
        #     colorscale='Viridis',
        #     colorbar=dict(title='Displacement Magnitude')
        #     )
        # )

        # # Create the layout
        # layout = go.Layout(
        #     title=f'Displacement Magnitude at Time Step {step}',
        #     scene=dict(
        #     xaxis_title='X',
        #     yaxis_title='Y',
        #     zaxis_title='Z'
        #     )
        # )

        # # Create the figure and plot it
        # fig = go.Figure(data=[scatter], layout=layout)
        
        # if step == 0 or step == num_steps - 1:
        #     fig.show()
        
            
        # Plot the node positions of the mesh for the first and last time step
        if step == 0 or step == num_steps - 1:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Extract node coordinates
            node_coords = mesh.geometry.x

            # Plot the nodes
            ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], c='b', marker='o', label='Initial Position' if step == 0 else 'Final Position')

            # If it's the last step, compare with initial positions
            if step == num_steps - 1:
                initial_coords = initial_node_coords
                moved_nodes = np.linalg.norm(node_coords - initial_coords, axis=1) > 1e-9
                ax.scatter(node_coords[moved_nodes, 0], node_coords[moved_nodes, 1], node_coords[moved_nodes, 2], c='r', marker='o', label='Moved Nodes')

            
            # Flip the z coordinates for the plot
        flipped_node_coords = node_coords.copy()
        flipped_node_coords[:, 2] = -flipped_node_coords[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the nodes with flipped z coordinates
        ax.scatter(flipped_node_coords[:, 0], flipped_node_coords[:, 1], flipped_node_coords[:, 2], c='b', marker='o', label='Initial Position' if step == 0 else 'Final Position')

        # If it's the last step, compare with initial positions
        if step == num_steps - 1:
            initial_flipped_coords = initial_node_coords.copy()
            initial_flipped_coords[:, 2] = -initial_flipped_coords[:, 2]
            moved_nodes = np.linalg.norm(flipped_node_coords - initial_flipped_coords, axis=1) > 1e-9
            ax.scatter(flipped_node_coords[moved_nodes, 0], flipped_node_coords[moved_nodes, 1], flipped_node_coords[moved_nodes, 2], c='r', marker='o', label='Moved Nodes')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.title(f'Node Positions at Time Step {step} (Z Flipped)')
            plt.show()
            

        # Store initial node coordinates for comparison
        if step == 0:
            initial_node_coords = mesh.geometry.x.copy()
            

        if comm.rank==0:
            print(f"Time step {step} done. top node 0 coords = {mesh.geometry.x[top_nodes[0]]}")
            print("Damage sample:", D.x.array[:5])

    if comm.rank==0:
        print("Done all steps.")


if __name__=="__main__":
    main_time_loop()
