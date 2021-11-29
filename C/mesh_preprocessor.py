import numpy as np
import trimesh.creation
import trimesh as tri

mesh = trimesh.creation.icosphere()
#mesh = tri.load_mesh("data.obj")

print(mesh.triangles)

triangles = np.array(mesh.triangles, dtype="double")

print(triangles)
print(len(triangles))

with open("triangles.num", "w+") as file:
    file.write(str(len(triangles.flatten())))

with open("triangles.data", "wb+") as file:
    file.write(triangles.tobytes())
