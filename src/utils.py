import numpy as np


def count_triangles_and_quads(polygonal_face_list):
    num_triangles = 0
    num_quads = 0

    for face in polygonal_face_list:
        num_vertices = len(face)
        if num_vertices == 3:
            num_triangles += 1
        elif num_vertices == 4:
            num_quads += 1
    return num_triangles, num_quads


def get_barycenter(vertex_matrix):
    return np.round(np.mean(vertex_matrix, axis=0), 3)


def get_principal_components(vertex_matrix):
    covariance_matrix = np.cov(np.transpose(vertex_matrix))
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    principal_components = [(val, vector) for val, vector in zip(eigenvalues, eigenvectors)]
    principal_components.sort(key=lambda x: x[0], reverse=False)
    return principal_components


def dot(a, b):
    return sum([a[i] * b[i] for i in range(len(a))])


def sign(n):
    if n < 0:
        return -1
    if n > 0:
        return 1
    return 0


def calculate_volume(vertex_matrix, face_matrix):
    overall_volume = 0
    for face_indices in face_matrix:

        v0, v1, v2 = vertex_matrix[face_indices]
        volume = abs(dot(v0, np.cross(v1, v2))) / 6
        overall_volume += volume
    return overall_volume


def calculate_face_area(face_matrix, vertex_matrix):
    list = []
    for face_indices in face_matrix:
        v0, v1, v2 = vertex_matrix[face_indices]
        list.append(abs(dot(v0, np.cross(v1, v2))) / 2)
    return list