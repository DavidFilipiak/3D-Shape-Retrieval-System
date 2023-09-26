import math
import os
import numpy as np

import pymeshlab
from pymeshlab import AbsoluteValue
from scipy.spatial.transform import Rotation as R
from utils import get_barycenter, dot
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
    R = np.dot(U, Vt)

    # Ensure that R has a valid determinant (to prevent reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(U, Vt)

    return R

def align(mesh: Mesh, meshSet: pymeshlab.MeshSet) -> Mesh:
    print("ORIGINAL MESH")
    vertex_matrix = meshSet.current_mesh().vertex_matrix()
    print(vertex_matrix.shape, vertex_matrix)
    covariance_matrix = np.cov(np.transpose(vertex_matrix))  # transpose, so that we get a 3x3 instead of nxn matrix
    print(covariance_matrix.shape, covariance_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print(eigenvalues.shape, eigenvalues)
    print(eigenvectors.shape, eigenvectors)

    principal_components = [(val, vector) for val, vector in zip(eigenvalues, eigenvectors)]
    principal_components.sort(key=lambda x: x[0], reverse=True)
    print("principal components", principal_components)
    rot_matrix_x_axis = align_vectors(principal_components[0][1], np.array([1, 0, 0]))
    print(rot_matrix_x_axis.dot(principal_components[0][1]))
    rot_matrix_x_axis = np.pad(rot_matrix_x_axis, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    rot_matrix_x_axis[3, 3] = 1

    rot_matrix_y_axis = align_vectors(principal_components[1][1], np.array([0, 1, 0]))
    print(rot_matrix_y_axis.dot(principal_components[1][1]))
    rot_matrix_y_axis = np.pad(rot_matrix_y_axis, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    rot_matrix_y_axis[3, 3] = 1

    rot_matrix_z_axis = align_vectors(principal_components[2][1], np.array([0, 0, 1]))
    print(rot_matrix_z_axis.dot(principal_components[2][1]))
    rot_matrix_z_axis = np.pad(rot_matrix_z_axis, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    rot_matrix_z_axis[3, 3] = 1

    #rot_matrix = rot_matrix_x_axis + rot_matrix_y_axis + rot_matrix_z_axis

    #print(rot_matrix_x_axis)
    #print(np.dot(rot_matrix_x_axis, principal_components[0][1]))
    #rot_matrix_y_axis = align_vectors(principal_components[1][1], np.array([0, 1, 0]))

    print('MY APPROACH MESH')
    #meshSet.set_matrix(transformmatrix=rot_matrix_x_axis)
    #meshSet.set_matrix(transformmatrix=rot_matrix_x_axis)
    #meshSet.set_matrix(transformmatrix=rot_matrix_z_axis)

    updated_coords = np.ndarray((len(vertex_matrix), 3))
    barycenter = get_barycenter(vertex_matrix)
    print("barycenter", barycenter)
    print("len e1", np.linalg.norm(principal_components[0][1]))
    print("len e2", np.linalg.norm(principal_components[1][1]))
    for i, vertex in enumerate(vertex_matrix):
        x = dot(vertex - barycenter, eigenvectors[0])
        y = dot(vertex - barycenter, eigenvectors[1])
        z = dot(vertex - barycenter, np.cross(eigenvectors[0], eigenvectors[1]))
        updated_coords[i] = np.array([x, y, z])
        print(updated_coords[i])

    rotation_matrix = find_rotation_matrix(updated_coords, vertex_matrix)
    rotation_matrix = np.pad(rotation_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    print("rotation matrix", rotation_matrix)
    rotation_matrix[3, 3] = 1
    meshSet.set_matrix(transformmatrix=rotation_matrix)

    print("dot", np.transpose(updated_coords).dot(vertex_matrix))
    covariance_matrix = np.cov(np.transpose(updated_coords))  # transpose, so that we get a 3x3 instead of nxn matrix
    print(covariance_matrix.shape, covariance_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print(eigenvalues.shape, eigenvalues)
    print(eigenvectors.shape, eigenvectors)

    """
    def get_dir_angles_of_vector(vector, axis):
        zeros = np.zeros(len(vector))
        zeros[axis] = 1
        print("dot product", np.dot(vector, zeros))
        print("norms", np.linalg.norm(vector), np.linalg.norm(zeros))
        return math.acos(np.dot(vector, zeros) / (np.linalg.norm(vector) * np.linalg.norm(zeros)))
    x_angle = get_dir_angles_of_vector(principal_components[0][1], 0)
    y_angle = get_dir_angles_of_vector(principal_components[1][1], 1)
    z_angle = get_dir_angles_of_vector(principal_components[2][1], 2)
    print(x_angle, y_angle, z_angle)

    r_x = R.from_rotvec(x_angle * np.array([0, 1, 1])).as_matrix()
    r_y = R.from_rotvec(y_angle * np.array([0, 1, 0])).as_matrix()
    r_z = R.from_rotvec(z_angle * np.array([0, 0, 1])).as_matrix()

    r_x = np.pad(r_x, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    print("r_x", r_x)
    r_y = np.pad(r_y, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    r_z = np.pad(r_z, ((0, 1), (0, 1)), mode='constant', constant_values=0)

    meshSet.set_matrix(transformmatrix=r_x)
    #meshSet.set_matrix(transformmatrix=r_y)
    #meshSet.set_matrix(transformmatrix=r_z)

    #transform_matrix = np.identity(4)
    #transform_matrix[:3, :3] = np.matmul(np.matmul(r_x, r_y), r_z)
    #print(transform_matrix)

    #meshSet.set_matrix(transformmatrix=transform_matrix)
    """
    """
    x_angle = np.degrees(x_angle)
    y_angle = np.degrees(y_angle)
    z_angle = np.degrees(z_angle)
    print(x_angle, y_angle, z_angle)

    meshSet.compute_matrix_from_rotation(angle=x_angle, rotcenter='barycenter', rotaxis='X axis')
    meshSet.compute_matrix_from_rotation(angle=y_angle, rotcenter='barycenter', rotaxis='Y axis')
    #meshSet.compute_matrix_from_rotation(angle=z_angle, rotcenter='barycenter', rotaxis='Z axis')
    
    #meshSet.compute_matrix_by_principal_axis()
    vertex_matrix = meshSet.current_mesh().vertex_matrix()
    covariance_matrix = np.cov(np.transpose(vertex_matrix))  # transpose, so that we get a 3x3 instead of nxn matrix
    print(covariance_matrix.shape, covariance_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print(eigenvalues.shape, eigenvalues)
    print(eigenvectors.shape, eigenvectors)
    """

    return mesh
