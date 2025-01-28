import gmsh
import meshio

def create_3D_extruded_mesh(Lx=1.0, Ly=1.0, Lz = 10.0, nx=5, ny=5, nz = 5,
                            outname="extruded_block.msh"):
    """
    Create a 2D rectangular region of size (Lx x Ly), meshed with nx x ny quads,
    and then extrude it in the z-direction by Lx (height = Lx).
    The result is a 3D block of hexahedra with dimension Lx x Ly x Lx.
    
    Arguments:
      Lx, Ly   : the rectangle dimensions in the x- and y-directions
      nx, ny   : number of cells (subdivisions) along x and y
      outname  : name of the .msh file to save
    """
    
    gmsh.initialize()
    gmsh.model.add("ExtrudedBlock")

    # ------------------------------------------------------------------------
    # 1) Define the 2D rectangle in the xy-plane
    # ------------------------------------------------------------------------
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
    p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0)
    p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0)
    
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    loop_id = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf_id = gmsh.model.geo.addPlaneSurface([loop_id])

    # ------------------------------------------------------------------------
    # 2) Make the 2D region transfinite & recombine for quads
    # ------------------------------------------------------------------------
    # Transfinite lines (ensures a structured distribution of nodes)
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny)

    # Transfinite surface using the corner points
    gmsh.model.geo.mesh.setTransfiniteSurface(surf_id, cornerTags=[p1, p2, p3, p4])

    # Synchronize and recombine to get quadrilaterals
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setRecombine(2, surf_id)

    # ------------------------------------------------------------------------
    # 3) Extrude the 2D surface to make a 3D block (hexahedra)
    #    Extrusion height is Lx for demonstration (you can change as needed).
    # ------------------------------------------------------------------------
    extrude_height = Lz

    # The `extrude` call returns a list of new entities generated:
    #   [(2, top_surface_id), (3, new_volume_id), ...]
    # We use numElements=[nz] if we want additional layers in z.
    # Here we match the 2D dimension for a "square cross-section" in 3D.  
    gmsh.model.geo.extrude(
        [(2, surf_id)],       # entities to extrude: (dimension=2, tag=surf_id)
        0, 0, extrude_height, # dx, dy, dz
        numElements=[nz - 1],     # number of subdivisions in the extruded direction
        recombine=True        # generate hexes instead of tetrahedra
    )
            
    gmsh.model.geo.synchronize()
    
    gmsh.model.mesh.setRecombine(2, surf_id)


    # for (etype, etag) in out:
    #     if etype == 2:
    #         gmsh.model.mesh.setRecombine(2, etag)

    # Generate the 3D mesh
    gmsh.model.mesh.generate(3)
    
    # ------------------------------------------------------------------------
    # 4) Save and finalize
    # ------------------------------------------------------------------------
    gmsh.write(outname)
    gmsh.finalize()
    print(f"Mesh saved to {outname}")
    
    m = meshio.read(outname)
    
    hex_blocks = []
    for block in m.cells:
        # block is a CellBlock object
        ctype = block.type   # e.g. "hexahedron", "tetra", etc.
        cdata = block.data   # NumPy array of shape (#cells, nodes_per_cell)
        if ctype == "hexahedron":
            hex_blocks.append(block)

    print("Number of hex cell blocks:", len(hex_blocks))

    # Create a new mesh object with only hex
    only_hex = meshio.Mesh(
        points=m.points,
        cells=hex_blocks,
        # you can also copy cell_data if relevant
    )
    
    meshio.write("extruded_block.xdmf", only_hex, file_format="xdmf")
    

if __name__ == "__main__":
    # Example usage: a 1 x 1 square, meshed into 5 x 5 quads, extruded by 1 in z
    # => results in a 1 x 1 x 1 cube of 5 x 5 x 5 hexahedra
    create_3D_extruded_mesh(Lx=2.0, Ly=4.0, Lz = 3.0, nx=3, ny=5, nz=4,
                            outname="extruded_block.msh")

