from tkinter import *
from tkinter import filedialog
import os
import pymeshlab
import pandas as pd
import numpy as np
import math
import polyscope as ps
from mesh import Mesh, meshes, feature_list
from database import Database
from matplotlib import pyplot as plt
from preprocess import normalize
from utils import count_triangles_and_quads
from pipeline import Pipeline

# GLOBAL VARIABLES
ms = None
database = None
listbox_loaded_meshes, label_loaded_meshes, current_mesh_label, current_csv_label = None, None, None, None
current_dir = os.getcwd()
selected_x = [0, 1000]  # example data for now, to store the list of values for x-axis
selected_y = [0, 2000]  # to store the list of values for y-axis
curr_mesh = None
load_files_recursive_counter = 0


def add_mesh_to_system(filename=""):
    global curr_mesh, current_mesh_label
    current_mesh = ms.current_mesh()
    mesh_name = "/".join(filename.split("/")[-2:])
    listbox_loaded_meshes.insert(END, mesh_name)
    num_triangles, num_quads = count_triangles_and_quads(current_mesh.polygonal_face_list())
    mesh = Mesh(current_mesh.id())
    mesh.set_params(
        num_vertices=current_mesh.vertex_number(),
        num_faces=current_mesh.face_number(),
        num_triangles=num_triangles,
        num_quads=num_quads,
        class_name=os.path.dirname(filename).split('/')[-1],
        name=mesh_name,
        bb_dim_x=current_mesh.bounding_box().dim_x(),
        bb_dim_y=current_mesh.bounding_box().dim_y(),
        bb_dim_z=current_mesh.bounding_box().dim_z(),
    )
    meshes[mesh.name] = mesh
    curr_mesh = mesh
    current_mesh_label.config(text=f"Current mesh: {mesh.name}")


def resample_mesh(mesh, vertex_num, face_num,filename) -> None:
    TARGET = 10000
    iter = 0
    best_iter = TARGET
    # Estimate number of faces to have 100+10000 vertex using Euler
    numFaces = 100 + 2 * TARGET
    simplification = False
    filename_save = filename.split("/")[-1].split('.')[0]
    last_slash_index = filename.rfind('/')
    result_path = filename[:last_slash_index]

    # Simplify the mesh. Only first simplification will be agressive
    while (mesh.current_mesh().vertex_number() > TARGET):
        simplification = True
        mesh.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces,
                          preservenormal=True)
        print("Decimated to", numFaces, "faces mesh has", mesh.current_mesh().vertex_number(), "vertex")
        # Refine our estimation to slowly converge to TARGET vertex number
        numFaces = numFaces - (mesh.current_mesh().vertex_number() - TARGET)
    else:
        while (mesh.current_mesh().vertex_number() <= TARGET):
            iter += 1
            mesh.meshing_surface_subdivision_butterfly(iterations=iter)
            print(f"vertice number {mesh.current_mesh().vertex_number()}")
            if (abs(mesh.current_mesh().vertex_number() - TARGET) < best_iter):
                best_iter = abs(mesh.current_mesh().vertex_number() - TARGET)
                mesh.save_current_mesh(os.path.join(result_path, filename_save + "_remeshed_increase.obj"))
            # else:
            #     mesh.meshing_surface_subdivision_butterfly(iterations=iter -1)
    print(f"vertice number {mesh.current_mesh().vertex_number()}")
    if (simplification):
        mesh.save_current_mesh(os.path.join(result_path, filename_save + "_resampled_decrease.obj"))


def browse_button() -> None:
    global curr_mesh
    db_dir = os.path.abspath(os.path.join(current_dir, "..", "db"))
    filename = filedialog.askopenfilename(title="Mesh select", initialdir=db_dir, filetypes=[('Mesh files', '*.obj')])
    ms.load_new_mesh(filename)
    add_mesh_to_system(filename)
    label_loaded_meshes.config(text=f"Loaded meshes ({len(ms)})")


def load_files_recursively(topdir, extension, limit=-1):
    global load_files_recursive_counter
    if limit == 0:
        return
    elif 0 < limit <= load_files_recursive_counter:
        return
    for root, dirs, files in os.walk(topdir):
        for file in files:
            if file.endswith(extension):
                ms.load_new_mesh(os.path.join(root, file))
                add_mesh_to_system(os.path.join(root.split("/")[-1], file))
                load_files_recursive_counter += 1
                print(f"Loaded {load_files_recursive_counter} meshes")
                if 0 < limit <= load_files_recursive_counter:
                    return
        for dir in dirs:
            load_files_recursively(os.path.join(root, dir), extension, limit)


def load_all_meshes_obj() -> None:
    global load_files_recursive_counter
    load_files_recursive_counter = 0
    folder_name = filedialog.askdirectory(title="Mesh select", initialdir=os.path.abspath(os.path.join(current_dir, "..", "db")))
    #load_files_recursively(folder_name, ".obj")
    for class_folder in os.listdir(folder_name):
        if os.path.isfile(os.path.join(folder_name, class_folder)):
            continue
        for file in os.listdir(os.path.join(folder_name, class_folder)):
            if file.endswith(".obj"):
                ms.load_new_mesh(os.path.join(folder_name, class_folder, file))
                add_mesh_to_system(os.path.join(class_folder, file))
                load_files_recursive_counter += 1
                print(f"Loaded {load_files_recursive_counter} meshes")
    label_loaded_meshes.config(text=f"Loaded meshes ({len(ms)})")


def load_all_meshes_csv() -> None:
    global current_csv_label
    filename = filedialog.askopenfilename(title="CSV select", initialdir=os.path.abspath(os.path.join(current_dir, "csv_files")), filetypes=[('CSV files', '*.csv')])
    database.load_table(filename)
    current_csv_label.config(text=f"Current CSV: {database.table_name}")


def save_current_mesh_obj() -> None:
    filename = filedialog.asksaveasfilename(title="Mesh save", initialdir=current_dir, filetypes=[('Mesh files', '*.obj')])
    ms.save_current_mesh(filename)


def save_all_meshes_csv() -> None:
    filename = filedialog.asksaveasfilename(title="CSV save", initialdir=current_dir, filetypes=[('CSV files', '*.csv')])
    feature_dict = {feature: [] for feature in feature_list}
    for mesh in meshes.values():
        f = mesh.get_features_dict()
        for feature in feature_list:
            feature_dict[feature].append(f[feature])
    df = pd.DataFrame(feature_dict)
    database.add_table(df, name=filename.split('/')[-1].split('.')[0])
    print(database.get_table)
    if not filename.endswith(".csv"):
        filename = filename + ".csv"
    database.save_table(filename)
    current_csv_label.config(text=f"Current CSV: {database.table_name}")


def show():
    ms.show_polyscope()


def normalize_btn():
    global ms, curr_mesh, meshes
    p = Pipeline(ms)
    p.add(normalize)
    normalized_meshes = p.run(list(meshes.values()))
    meshes = {mesh.name: mesh for mesh in normalized_meshes}
    print("Normalized")


# right now this function only loads custom features from the csv_files file until real ones will go there
def analyze_meshes() -> None:
    features_table = database.get_table()
    print(features_table)
    # TODO: Jesse --> analyze and print requested features, use function from pandas when possible
    analyze_df(features_table)


def analyze_df(df) -> None:
    for f in df:
        feature = pd.Series([f]).describe()  # More descriptive statistics should go in here
        print(feature)


def analyze_feature(feature):
    table = database.get_table()
    table.sort_values(feature, inplace=True)
    values = table[feature].values
    max_value = table[feature].max()
    row_with_max_value = table[table[feature] == max_value]
    min_value = table[feature].min()
    row_with_min_value = table[table[feature] == min_value]
    average_value = table[feature].mean()
    table['temp_abs_diff'] = abs(table[feature] - average_value)
    closest_avg_row = table[table['temp_abs_diff'] == table['temp_abs_diff'].min()]
    closest_avg_row = closest_avg_row.drop('temp_abs_diff', axis=1)[["name", feature]]
    table.drop('temp_abs_diff', axis=1, inplace=True)

    print("Analysis of Feature:", feature)
    print("Max value:", max_value)
    print("Max value mesh:", row_with_max_value["name"].values[0], "with value:", row_with_max_value[feature].values[0])
    print("Min value:", min_value)
    print("Min value mesh:", row_with_min_value["name"].values[0], "with value:", row_with_min_value[feature].values[0])
    print("Average value:", average_value)
    print("Closest mesh:", closest_avg_row["name"].values[0], "with value:", closest_avg_row[feature].values[0])

    hist_y, hist_x = np.histogram(values, bins=math.ceil(math.sqrt(len(values))))
    if max_value - min_value > 10:
        hist_x = hist_x.astype(int)
    else:
        # round to 2 decimal places
        hist_x = np.round(hist_x, 2)
    histogram = draw_histogram(hist_x[:-1], hist_y)


def draw_histogram(arr_x, arr_y):
    plt.rcParams["figure.figsize"] = [13, 6]
    plt.rcParams["figure.autolayout"] = True
    width = np.mean(arr_x[1:] - arr_x[:-1]) / 2
    fig = plt.bar(arr_x, arr_y, width=width, color="blue", align='edge')
    plt.xticks([arr_x[i] for i in range(0, len(arr_x), 2) if arr_y[i] > 0])
    for i in range(1, len(arr_x), 2):
        if arr_y[i] > 0:
            plt.text(width * i * 2, arr_y[i], str(arr_x[i]), fontsize=10)
    plt.xlabel("Bin size")
    plt.ylabel("Number of meshes")
    plt.show()
    return fig


def do_nothing():
    pass


def main() -> None:
    global ms, listbox_loaded_meshes, curr_mesh, label_loaded_meshes, database, current_mesh_label, current_csv_label
    ms = pymeshlab.MeshSet()
    database = Database()

    root = Tk()
    root.title("3D Shape Retrieval")
    root.geometry("500x500")

    menubar = Menu(root)
    # File menu
    filemenu = Menu(menubar, tearoff=0)
    filemenu.add_command(label="Load Mesh (.obj)", command=browse_button)
    filemenu.add_command(label="Load All (.obj)", command=load_all_meshes_obj)
    filemenu.add_command(label="Load All (.csv)", command=load_all_meshes_csv)
    filemenu.add_separator()
    filemenu.add_command(label="Save Current Mesh (.obj)", command=save_current_mesh_obj)
    filemenu.add_command(label="Save All (.csv)", command=save_all_meshes_csv)
    menubar.add_cascade(label="File", menu=filemenu)
    # Show menu
    showmenu = Menu(menubar, tearoff=0)
    showmenu.add_command(label="Current Mesh", command=do_nothing)
    showmenu.add_command(label="Selected Meshes", command=do_nothing)
    showmenu.add_command(label="All Loaded Meshes", command=show)
    menubar.add_cascade(label="Show", menu=showmenu)
    # Analyze menu
    analyzemenu = Menu(menubar, tearoff=0)
    analyzemenu.add_command(label="Current Mesh", command=do_nothing)
    analyzemenu.add_separator()
    featuresmenu = Menu(analyzemenu, tearoff=0)
    for feature in feature_list:
        if feature not in ['name', 'class_name']:
            featuresmenu.add_command(label=feature, command=lambda f=feature: analyze_feature(f))
    analyzemenu.add_cascade(label="All Loaded Meshes (.csv)", menu=featuresmenu)
    menubar.add_cascade(label="Analyze", menu=analyzemenu)
    # Preprocess menu
    preprocessmenu = Menu(menubar, tearoff=0)
    preprocessmenu.add_command(label="Full", command=do_nothing)
    preprocessmenu.add_separator()
    preprocessmenu.add_command(label="Normalize", command=normalize_btn)
    preprocessmenu.add_command(label="Resample", command=do_nothing)
    menubar.add_cascade(label="Preprocess", menu=preprocessmenu)

    root.config(menu=menubar)

    current_mesh_label = Label(root, text="Current mesh")
    current_mesh_label.grid(row=0, column=0)
    current_csv_label = Label(root, text="Current CSV table")
    current_csv_label.grid(row=0, column=1)

    button_analyze = Button(text="Analyze", command=analyze_meshes)
    button_analyze.grid(row=0, column=3)

    label_loaded_meshes = Label(root, text="Loaded meshes")
    label_loaded_meshes.grid(row=1, column=0)
    listbox_loaded_meshes = Listbox(root, width=50)
    listbox_loaded_meshes.grid(row=2, column=0, columnspan=3)

    #button_graph = Button(root, text="Show histogram", command=draw_histogram(selected_x, selected_y))
    #button_graph.grid(row=3, column=1)

    root.mainloop()


if __name__ == "__main__":
    main()