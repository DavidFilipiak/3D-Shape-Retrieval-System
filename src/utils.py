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

def dot(a, b):
    return sum([a[i] * b[i] for i in range(len(a))])

def sign(n):
    if n < 0:
        return -1
    if n > 0:
        return 1
    return 0
