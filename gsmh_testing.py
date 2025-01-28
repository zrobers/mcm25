import gmsh

gmsh.initialize()

# Create a square
lc = 0.1  # Characteristic length
p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
p2 = gmsh.model.geo.addPoint(1, 0, 0, lc)
p3 = gmsh.model.geo.addPoint(1, 1, 0, lc)
p4 = gmsh.model.geo.addPoint(0, 1, 0, lc)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
surface = gmsh.model.geo.addPlaneSurface([loop])


gmsh.option.setNumber("Mesh.Algorithm", 8)  # Use "Transfinite" algorithm for structured mesh
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)  # Improve mesh quality 
gmsh.model.mesh.setRecombine(surface, 0)

# Extrude the surface 2 units in the z-direction
extruded = gmsh.model.geo.extrude([(2, surface)], 0, 0, 0.5)

gmsh.model.geo.synchronize()


# Generate the mesh
gmsh.model.mesh.generate(3)

# Launch the GUI to view the mesh
gmsh.fltk.run()

gmsh.finalize()