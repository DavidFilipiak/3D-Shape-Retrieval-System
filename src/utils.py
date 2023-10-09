import numpy as np
from matplotlib import pyplot as plt

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


def get_mass_directions(vertex_matrix, face_matrix):
    face_centres = np.ndarray((len(face_matrix), 3))
    for i, face in enumerate(face_matrix):
        xs, ys, zs = [], [], []
        for v in face:
            xs.append(vertex_matrix[v][0])
            ys.append(vertex_matrix[v][1])
            zs.append(vertex_matrix[v][2])
        x = np.mean(xs)
        y = np.mean(ys)
        z = np.mean(zs)
        face_centres[i] = np.array([x, y, z])

    fx, fy, fz = 0, 0, 0
    for center in face_centres:
        fx += sign(center[0]) * center[0] ** 2
        fy += sign(center[1]) * center[1] ** 2
        fz += sign(center[2]) * center[2] ** 2

    return np.array([sign(fx), sign(fy), sign(fz)])


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
        list.append(abs(np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2))
    return list


def draw_histogram(arr_x, arr_y, min, max, mean=None, std=None, xlabel="Bin size", ylabel="Number of meshes"):
    plt.rcParams["figure.figsize"] = [13, 6]
    plt.rcParams["figure.autolayout"] = True
    plt.xlim(min, max)

    if np.min(arr_x) > min:
        min = np.min(arr_x)
    if np.max(arr_x) < max:
        max = np.max(arr_x)

    width = (max - min) / len(arr_x)

    print(width, max, min, len(arr_x))
    fig = plt.bar(arr_x, arr_y, width=width, color="blue", align='edge')
    if mean is not None:
        plt.axvline(mean, color='black', linestyle='dashed', linewidth=1)
    if std is not None:
        plt.axvline(mean - std, color='grey', linestyle='dashed', linewidth=0.7)
        plt.axvline(mean + std, color='grey', linestyle='dashed', linewidth=0.7)
    #plt.xticks([arr_x[i] for i in range(0, len(arr_x), 2) if arr_y[i] > 0])
    #for i in range(1, len(arr_x), 2):
    #    if arr_y[i] > 0:
    #        plt.text(width * i * 2, arr_y[i], str(arr_x[i]), fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return fig


def draw_grouped_histogram(arr_x, arrs_y, x_label="Bin size", y_label="Number of meshes"):
    plt.rcParams["figure.figsize"] = [13, 6]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    width = max(arr_x) / (len(arr_x) * len(arrs_y[0]) * (max(arr_x) - min(arr_x)))
    x = np.arange(len(arr_x))
    for i, arr_y in enumerate(arrs_y):
        ax.bar(x + i * width, arr_y, width=width, align='edge')
    ax.set_xticks(x + (width * 1.5))
    ax.set_xticklabels(arr_x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.legend(["x", "y", "z"], loc='upper left', ncols=3)
    plt.show()
    return fig