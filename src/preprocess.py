import math
import os

import pymeshlab
from pymeshlab import AbsoluteValue

from mesh import Mesh, meshes
'''
REMESH -  if vertices > TARGET reduce them
          if vertices <= TARGET reduce them
'''

def resample_mesh(mesh: Mesh, meshSet: pymeshlab.MeshSet, result_filename = '') -> Mesh:
    TARGET = 10000
    iter = 0
    # Estimate number of faces to have 100+10000 vertex using Euler
    numFaces = 100 + 2 * TARGET
    target_edge_length = 0.001
    previous_vertex_count = None
    consecutive_constant_count = 0
    max_consecutive_constant_iterations = 1
    while (meshSet.current_mesh().vertex_number() <= 10*TARGET):
        meshSet.meshing_isotropic_explicit_remeshing(targetlen=AbsoluteValue(target_edge_length), iterations=iter)
        current_vertex_count = meshSet.current_mesh().vertex_number()
        print(f"Vertex number: {current_vertex_count}")
        # Check if the current vertex count is the same as the previous vertex count
        if current_vertex_count == previous_vertex_count:
            consecutive_constant_count += 1
        else:
            consecutive_constant_count = 0
        # If the vertex count has remained constant for two iterations, break the loop
        if consecutive_constant_count >= max_consecutive_constant_iterations:
            print('breaking from loop')
            break

        # Update the previous vertex count
        previous_vertex_count = current_vertex_count
        # Simplify the mesh. Only first simplification will be agressive
    while (meshSet.current_mesh().vertex_number() > TARGET):
        meshSet.meshing_repair_non_manifold_edges()
        print(meshSet.current_mesh().label())
        meshSet.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=numFaces,
                             preservenormal=True)
        print("Decimated to", numFaces, "faces mesh has", meshSet.current_mesh().vertex_number(), "vertex")
        # Refine our estimation to slowly converge to TARGET vertex number
        numFaces = numFaces - (meshSet.current_mesh().vertex_number() - TARGET)
    meshSet.save_current_mesh(os.path.join(result_filename,mesh.name.split('.')[0] + "_resampled.obj"))
    current_mesh = meshSet.current_mesh()
    mesh.set_params(
        num_faces =current_mesh.vertex_number(),
        num_vertices=current_mesh.face_number()
    )
    return mesh



def resample_mesh_david_attempt(mesh: Mesh, meshSet: pymeshlab.MeshSet, result_filename = '') -> Mesh:
    TARGET_LOW = 8000
    TARGET_HIGH = 13000
    iter = 0

    target_edge_length = max(mesh.bb_dim_x, mesh.bb_dim_y, mesh.bb_dim_z) / 100
    previous_vertex_count = meshSet.current_mesh().vertex_number()

    while not (TARGET_LOW <= meshSet.current_mesh().vertex_number() <= TARGET_HIGH):
        iter += 1
        print(f"iteration {iter}, target_edge_length {target_edge_length}")
        # Do just one remeshing iteration per one while iteration
        meshSet.meshing_isotropic_explicit_remeshing(targetlen=AbsoluteValue(target_edge_length), iterations=1)
        # meshSet.repair_non_manifold_edges()
        print(f"Vertex number: {previous_vertex_count}")
        current_vertex_count = meshSet.current_mesh().vertex_number()
        print(f"Vertex number: {current_vertex_count}")
        # Check if the current vertex count is the same as the previous vertex count
        var = max(0.0, math.log10(current_vertex_count) - 2)
        if current_vertex_count < TARGET_LOW or current_vertex_count > TARGET_HIGH:
            gap = 1000
        else:
            gap = 100
        #print(gap)
        rate_of_change = 1/10
        diff_low, diff_high = current_vertex_count - TARGET_LOW, TARGET_HIGH - current_vertex_count


        if current_vertex_count - gap <= previous_vertex_count <= current_vertex_count + gap:
            if current_vertex_count < TARGET_LOW:
                print("IF 1")
                target_edge_length *= (1 - rate_of_change)
            elif current_vertex_count > TARGET_HIGH:
                print("IF 2")
                target_edge_length *= (1 + rate_of_change)
            else:
                print("IF 3")
        else:
            print("IF 4")

        # Update the previous vertex count
        previous_vertex_count = current_vertex_count

    current_mesh = meshSet.current_mesh()
    mesh.set_params(
        num_faces =current_mesh.vertex_number(),
        num_vertices=current_mesh.face_number()
    )
    return mesh


# translate into origin and scale to unit cube
def normalize(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:

    # apply filters
    meshSet.compute_matrix_from_translation(traslmethod='Center on Layer BBox', alllayers=True)
    meshSet.compute_matrix_from_scaling_or_normalization(unitflag=True, scalecenter='barycenter')

    # change parameters of current mesh
    current_mesh = meshSet.current_mesh()
    mesh.set_params(
        bb_dim_x=current_mesh.bounding_box().dim_x(),
        bb_dim_y=current_mesh.bounding_box().dim_y(),
        bb_dim_z=current_mesh.bounding_box().dim_z(),
    )
    return mesh
