import math
import os
import numpy as np
import pymeshlab
from matplotlib import pyplot as plt
from pymeshlab import AbsoluteValue

from utils import *
from mesh import Mesh
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
    while (meshSet.current_mesh().vertex_number() <= TARGET):
        iter += 1
        meshSet.meshing_isotropic_explicit_remeshing(targetlen=AbsoluteValue(target_edge_length), iterations=iter)
        print(f"vertice number {meshSet.current_mesh().vertex_number()}")
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
        # meshSet.meshing_repair_non_manifold_edges()
        print(meshSet.current_mesh().label())
        meshSet.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=numFaces,
                             preservenormal=True)
        print("Decimated to", numFaces, "faces mesh has", meshSet.current_mesh().vertex_number(), "vertex")
        # Refine our estimation to slowly converge to TARGET vertex number
        numFaces = numFaces - (meshSet.current_mesh().vertex_number() - TARGET)
    current_mesh = meshSet.current_mesh()
    mesh.set_params(
        num_faces =current_mesh.face_number(),
        num_vertices=current_mesh.vertex_number()
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
        num_faces=current_mesh.vertex_number(),
        num_vertices=current_mesh.face_number()
    )
    return mesh


def translate_to_origin(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    # apply filters
    # meshSet.compute_matrix_from_translation(traslmethod='Center on Layer BBox', alllayers=True)

    barycenter = get_barycenter(meshSet.current_mesh().vertex_matrix())
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 3] = -barycenter

    meshSet.set_matrix(transformmatrix=transform_matrix)

    # change parameters of current mesh
    current_mesh = meshSet.current_mesh()
    mesh.set_params(
        bb_dim_x=current_mesh.bounding_box().dim_x(),
        bb_dim_y=current_mesh.bounding_box().dim_y(),
        bb_dim_z=current_mesh.bounding_box().dim_z(),
        bb_diagonal=current_mesh.bounding_box().diagonal(),
    )
    return mesh


def scale_to_unit_cube(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    # apply filters
    # meshSet.compute_matrix_from_scaling_or_normalization(unitflag=True, scalecenter='barycenter')

    bb = meshSet.current_mesh().bounding_box()
    min_point = bb.min()
    max_point = bb.max()
    scale = max(max_point[0] - min_point[0], max_point[1] - min_point[1], max_point[2] - min_point[2])
    transform_matrix = np.eye(4) * (1 / scale)
    transform_matrix[3, 3] = 1

    meshSet.set_matrix(transformmatrix=transform_matrix)

    # change parameters of current mesh
    current_mesh = meshSet.current_mesh()
    mesh.set_params(
        bb_dim_x=current_mesh.bounding_box().dim_x(),
        bb_dim_y=current_mesh.bounding_box().dim_y(),
        bb_dim_z=current_mesh.bounding_box().dim_z(),
    )
    return mesh


# https://stackoverflow.com/questions/67017134/find-rotation-matrix-to-align-two-vectors
# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
def align_vectors(a, b):
    print("a", a)
    print("b", b)

    b = b / np.linalg.norm(b)  # normalize a
    a = a / np.linalg.norm(a)  # normalize b
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)

    v1, v2, v3 = v
    h = (1 - c) / (s ** 2)

    Vmat = np.array([[0, -v3, v2],
                     [v3, 0, -v1],
                     [-v2, v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R

def find_rotation_matrix(A, B):
    # Center the points (subtract the mean)
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    centered_A = A - centroid_A
    centered_B = B - centroid_B

    # Calculate the covariance matrix H
    H = np.dot(centered_A.T, centered_B)

    # Use SVD to find the rotation matrix R
    U, _, Vt = np.linalg.svd(H)
    #return U.T @ Vt.T
    R = np.dot(U, Vt)

    # Ensure that R has a valid determinant (to prevent reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(U, Vt)

    return R

def align(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    meshSet.compute_matrix_by_principal_axis()
    current_mesh = meshSet.current_mesh()
    principal_components = get_principal_components(current_mesh.vertex_matrix())
    mesh.set_params(
        bb_dim_x=current_mesh.bounding_box().dim_x(),
        bb_dim_y=current_mesh.bounding_box().dim_y(),
        bb_dim_z=current_mesh.bounding_box().dim_z(),
        bb_diagonal=current_mesh.bounding_box().diagonal(),
        major_eigenvector=principal_components[0][1],
        median_eigenvector=principal_components[1][1],
        minor_eigenvector=principal_components[2][1]
    )
    return mesh


def flip(mesh:Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    vertex_matrix = meshSet.current_mesh().vertex_matrix()
    face_matrix = meshSet.current_mesh().face_matrix()

    x, y, z = get_mass_directions(vertex_matrix, face_matrix)
    transform_matrix = np.eye(4)
    transform_matrix[0, 0] = x
    transform_matrix[1, 1] = y
    transform_matrix[2, 2] = z

    meshSet.set_matrix(transformmatrix=transform_matrix)

    mesh.set_params(
        mass_directions=np.array([x, y, z])
    )

    return mesh
