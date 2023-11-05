import numpy as np
from matplotlib import pyplot as plt
import mplcursors
import random
import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost
import itertools

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
    principal_components.sort(key=lambda x: x[0], reverse=True)
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

def flatten_nested_array(arr):
    flattened_elements = []
    for element in arr:
        if isinstance(element, np.ndarray):
            flattened_elements.extend(list(element))
        else:
            flattened_elements.append(element)
    resulting_array = np.array(flattened_elements)
    return resulting_array

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
        volume = abs(dot(v0, np.cross(v1,v2)))/ 6
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

    width = ((max - min) / len(arr_x))

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

def draw_line_histograms(class_histograms, x_label="", y_label="", line_limit=15):
    plt.rcParams["figure.figsize"] = [13, 6]
    plt.rcParams["figure.autolayout"] = True
    num_plots = len(class_histograms)
    square = reshape_to_square_matrix(np.arange(num_plots))
    fig = plt.figure(1)
    plt.title(x_label)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    for i in range(square.shape[0]):
        for j in range(square.shape[1]):
            if square[i][j] > -1:
                index = int(square[i][j])
                ax = fig.add_subplot(square.shape[0], square.shape[1], index + 1)
                class_name, histograms = class_histograms[index]
                random.shuffle(histograms)
                for count in range(min(line_limit, len(histograms))):
                    histogram = histograms[count]
                    arr_x, arr_y = histogram[0], histogram[1]
                    color = np.random.rand(3, )
                    ax.plot(arr_x, arr_y, color=color)
                    plt.xlabel(class_name)

    plt.show()
    return fig


def reshape_to_square_matrix(arr):
    # Calculate the size of the square matrix
    size = int(np.ceil(np.sqrt(len(arr))))

    # Calculate the number of NaN values needed to fill the matrix
    num_nan = size * size - len(arr)

    # Create a new matrix filled with NaN values
    matrix = np.empty((size, size))
    matrix[:] = -1

    # Fill the matrix with values from the original array
    for i in range(len(arr)):
        matrix[i // size][i % size] = arr[i]

    return matrix


def draw_scatterplot(df_data, x_label="", y_label="", title=""):
    plt.rcParams["figure.figsize"] = [13, 6]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    df_data = df_data.sort_values(by=['name', 'class_name'])
    classes = df_data['class_name'].unique()
    names = df_data['name'].values
    distinct_colors = ["#2f4f4f","#800000","#008000","#4b0082","#ff8c00","#ffff00","#00ff00","#00ffff","#0000ff","#ff00ff","#eee8aa","#6495ed","#ff69b4"]
    distinct_shapes = ["o", "v", "^", "s", "<", ">"]
    #classes = random.sample(list(classes), 20)
    #classes = ["HumanHead", "Helicopter"]
    scatter = None
    show_classes = ["Bicycle", "Car", "Fish", "HumanHead"]
    #show_classes = classes
    for i, class_name in enumerate(classes):
        data = df_data.loc[df_data['class_name'] == class_name].iloc[:, 2:].values
        names = df_data.loc[df_data['class_name'] == class_name].iloc[:, 0].values
        color = distinct_colors[i % len(distinct_colors)]
        shape = distinct_shapes[i % len(distinct_shapes)]
        #with open('legend.txt', 'a') as f:
        #    f.write(class_name + "," + color + "," + shape + "\n")

        if class_name in show_classes:
            ax.scatter(data[:, 0], data[:, 1], s=10, color=color, marker=shape, label=names, edgecolors='black', linewidths=0.1)


    mplcursors.cursor(ax.artists, hover=2).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
        #mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(classes[sel.target.index]))

    #plt.legend(classes, loc='upper left', ncols=3, bbox_to_anchor=(1, 1))
    plt.legend(show_classes, loc='upper left', ncols=3, bbox_to_anchor=(1, 1))
    plt.show()
    return fig


def draw_heatmap(matrix, class2index):
    plt.rcParams["figure.figsize"] = [6, 6]
    plt.rcParams["figure.autolayout"] = True

    fig = plt.figure()
    ax = SubplotHost(fig, 111)
    fig.add_subplot(ax)

    for class_name, (start, end) in class2index.items():
        ax.axvline(start, color='black', linestyle='solid', linewidth=0.5)
        ax.axhline(start, color='black', linestyle='solid', linewidth=0.5)
        ax.axvline(end, color='black', linestyle='solid', linewidth=0.5)
        ax.axhline(end, color='black', linestyle='solid', linewidth=0.5)
    im = ax.imshow(matrix, norm="symlog")
    plt.colorbar(im)
    ax.tick_params(left=False, bottom=False, right=False, top=False, labelleft=False, labelbottom=False, labelright=False, labeltop=False)
    # Second X-axis

    ax2 = ax.twiny()
    offset = 0, -25  # Position of the second axis
    new_axisline = ax2.get_grid_helper().new_fixed_axis
    ax2.axis["bottom"] = new_axisline(loc="bottom", axes=ax2, offset=(0,-1))
    #ax2.axis["left"] = new_axisline(loc="left", axes=ax2, offset=(-25, 0))
    ax2.axis["top"].set_visible(False)

    ax2.set_xticks([start for start, _ in class2index.values()] + [list(class2index.values())[-1][1]])
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator([start + (end - start) / 2 for start, end in class2index.values()]))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(list(class2index.keys())))
    ax2.tick_params(axis='x', which='both', labelsize=100, labelrotation=90)


    #ax.grid(1)
    plt.show()
    return fig
