from tkinter import *
from tkinter import filedialog
import os
import pymeshlab
import pandas as pd
import numpy as np
import math
import polyscope as ps
from mesh import Mesh, meshes
from feature import feature_list
from database import Database
from matplotlib import pyplot as plt
from preprocess import translate_to_origin, scale_to_unit_cube, resample_mesh, align, flip
from utils import *
from pipeline import Pipeline
from feature import *
# GLOBAL VARIABLES
ms = None
database = None
listbox_loaded_meshes, label_loaded_meshes, current_mesh_label, current_csv_label = None, None, None, None
current_dir = os.getcwd()
selected_x = [0, 1000]  # example data for now, to store the list of values for x-axis
selected_y = [0, 2000]  # to store the list of values for y-axis
curr_mesh = None


def add_mesh_to_system(filename=""):
    global curr_mesh, current_mesh_label
    current_mesh = ms.current_mesh()
    mesh_name = "/".join(filename.split("/")[-2:])
    listbox_loaded_meshes.insert(END, mesh_name)
    num_triangles, num_quads = count_triangles_and_quads(current_mesh.polygonal_face_list())
    mesh = Mesh(current_mesh.id())
    ###MOST OF THE FEATURES SHOULD BE EXTRACTED AFTER THE PREPROCESSING
    bb = current_mesh.bounding_box()
    min_point = bb.min()
    max_point = bb.max()
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
        bb_diagonal=current_mesh.bounding_box().diagonal(),
        barycenter=get_barycenter(current_mesh.vertex_matrix())
        #volume=out_dict_geom['mesh_volume'],
        #surface_area=out_dict_geom['surface_area'],
        #average_edge_length=out_dict_geom['avg_edge_length'],
        #total_edge_length=out_dict_geom['total_edge_length'],
        #center_of_mass=out_dict_geom["center_of_mass"],
        #connected_components_number=out_dict_top["connected_components_number"],
        #convex_hull=ms.generate_convex_hull(),
        #eccentricity=math.sqrt(1 - (scale_min/scale_long)**2)
    )
    meshes[mesh.name] = mesh
    curr_mesh = mesh
    current_mesh_label.config(text=f"Current mesh: {mesh.name}")



def browse_button() -> None:
    global curr_mesh
    db_dir = os.path.abspath(os.path.join(current_dir, "..", "db"))
    filename = filedialog.askopenfilename(title="Mesh select", initialdir=db_dir, filetypes=[('Mesh files', '*.obj')])
    ms.load_new_mesh(filename)
    add_mesh_to_system(filename)
    label_loaded_meshes.config(text=f"Loaded meshes ({len(ms)})")


def load_files_recursively(topdir, extension, limit=-1, offset=0, count=0) -> int:
    if limit == 0:
        return count
    elif 0 < limit <= count:
        return count

    root, dirs, files = list(os.walk(topdir))[0]
    for file in files:
        if file.endswith(extension):
            if 0 < limit <= count:
                return count
            if count < offset:
                count += 1
                continue
            new_path = os.path.join(root, file).replace("\\", "/")
            ms.load_new_mesh(new_path)
            add_mesh_to_system(new_path)
            count += 1

    for _dir in dirs:
        count = load_files_recursively(os.path.join(root, _dir), extension, limit, offset, count)
    return count


def load_all_meshes_obj() -> None:
    LIMIT = -1
    folder_name = filedialog.askdirectory(title="Mesh select", initialdir=os.path.abspath(os.path.join(current_dir, "..", "db")))
    count = load_files_recursively(folder_name, ".obj", limit=LIMIT)
    if count != len(ms):
        raise Exception(f"A problem while loading meshes.")
    label_loaded_meshes.config(text=f"Loaded meshes ({len(ms)})")


def load_all_meshes_csv() -> None:
    global current_csv_label
    filename = filedialog.askopenfilename(title="CSV select", initialdir=os.path.abspath(os.path.join(current_dir, "csv_files")), filetypes=[('CSV files', '*.csv')])
    database.load_table(filename)
    current_csv_label.config(text=f"Current CSV: {database.table_name}")

def clear_all_meshes_obj() -> None:
    database.clear_table()
    ms.clear()
    label_loaded_meshes.config(text=f"Loaded meshes ({len(ms)})")
    listbox_loaded_meshes.delete(0,END)

def clear_selected_meshes_obj() -> None:
    global ms,data
    for mesh in ms:
        if mesh.label() == data.split('/')[-1]:
            ms.set_current_mesh(mesh.id())
            ms.delete_current_mesh()
            listbox_loaded_meshes.delete(ACTIVE)
            break
    # ms.delete_current_mesh()
    label_loaded_meshes.config(text=f"Loaded meshes ({len(ms)})")

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


def do_translate():
    global ms, curr_mesh, meshes
    p = Pipeline(ms)
    p.add(translate_to_origin)
    normalized_meshes = p.run(list(meshes.values()))
    meshes = {mesh.name: mesh for mesh in normalized_meshes}
    print("Moved to origin")


def do_scale():
    global meshes
    p = Pipeline(ms)
    p.add(scale_to_unit_cube)
    normalized_meshes = p.run(list(meshes.values()))
    meshes = {mesh.name: mesh for mesh in normalized_meshes}
    print("Scaled to unit cube")


def batch_preprocess():
    global meshes
    output_dir = os.path.abspath(os.path.join(current_dir, "..", "preprocessed"))
    folder_name = filedialog.askdirectory(title="Mesh select", initialdir=os.path.abspath(os.path.join(current_dir, "..", "db")))
    batch_size = 1
    batch_offset = 0
    pipeline = Pipeline(ms)
    pipeline.add(resample_mesh)
    pipeline.add(translate_to_origin)
    pipeline.add(align)
    pipeline.add(flip)
    pipeline.add(scale_to_unit_cube)
   # pipeline.add(resample_mesh_david_attempt)

    file_count = load_files_recursively(folder_name, ".obj", limit=batch_size, offset=batch_offset)
    while file_count == batch_size:
        batch_offset += batch_size
        pipeline.run(list(meshes.values()))
        for mesh in meshes.values():
            ms.set_current_mesh(mesh.pymeshlab_id)
            if not os.path.exists(os.path.join(output_dir, mesh.name.split("/")[0])):
                os.mkdir(os.path.join(output_dir, mesh.name.split("/")[0]))
            # calculate surface area
            Surface_area.value = ms.get_geometric_measures()['surface_area']
            # calculate volume
            Volume.value = calculate_volume(ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix())
            # calculate compactness
            Compactness.value = (36 * math.pi * Volume.value ** 2) ** (1 / 3) / Surface_area.value
            # calculate Axis aligned bounding box
            AABB_volume.value = ms.current_mesh().bounding_box().dim_x() * ms.current_mesh().bounding_box().dim_y() * ms.current_mesh().bounding_box().dim_z()
            # calculate eccentricity
            scale_long = max(ms.current_mesh().bounding_box().dim_x(), ms.current_mesh().bounding_box().dim_y(),
                             ms.current_mesh().bounding_box().dim_z())
            scale_min = min(ms.current_mesh().bounding_box().dim_x(), ms.current_mesh().bounding_box().dim_y(),
                            ms.current_mesh().bounding_box().dim_z())
            Eccentricity.value = math.sqrt(1 - (scale_min / scale_long) ** 2)
            # calculate rectangularity
            Rectangularity.value = AABB_volume.value / Volume.value
            # calculate convexivity
            Convexivity.value = Volume.value / AABB_volume.value
            # find longest distance between two points
            distances = np.linalg.norm(ms.current_mesh().vertex_matrix() - ms.current_mesh().vertex_matrix().mean(axis=0), axis=1)
            Diameter.value = distances.max()
            ms.save_current_mesh(os.path.join(output_dir, *(mesh.name.split("/"))))

        clear_all_meshes_obj()

        if batch_size == 10:
            break
        file_count = load_files_recursively(folder_name, ".obj", limit=batch_size, offset=batch_offset)



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


def do_resample():
    global ms, curr_mesh,meshes
    p = Pipeline(ms)
    rec_path = os.path.join(os.path.dirname(current_dir), 'db')
    p.add(resample_mesh, result_filename= rec_path)
    remeshed_meshes = p.run(list(meshes.values()))
    meshes = {mesh.name: mesh for mesh in remeshed_meshes}
    print("Remeshed")


def do_align():
    global meshes
    p = Pipeline(ms)
    p.add(align)
    aligned_meshes = p.run(list(meshes.values()))
    meshes = {mesh.name: mesh for mesh in aligned_meshes}
    print("Aligned")


def do_flip():
    global meshes
    p = Pipeline(ms)
    p.add(flip)
    aligned_meshes = p.run(list(meshes.values()))
    meshes = {mesh.name: mesh for mesh in aligned_meshes}
    print("Flipped")


def do_nothing():
    pass

def main() -> None:
    global ms, listbox_loaded_meshes, curr_mesh, label_loaded_meshes, database, current_mesh_label, current_csv_label,filename, data
    filename = ''
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
    filemenu.add_command(label="Clear All (.obj)", command= clear_all_meshes_obj)
    filemenu.add_command(label="Clear Selected (.obj)", command=clear_selected_meshes_obj)
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
    preprocessmenu.add_command(label="Batch", command=batch_preprocess)
    preprocessmenu.add_command(label="Full", command=do_nothing)
    preprocessmenu.add_separator()
    preprocessmenu.add_command(label="Resample", command=do_resample)
    preprocessmenu.add_command(label="Translate", command=do_translate)
    preprocessmenu.add_command(label="Align", command=do_align)
    preprocessmenu.add_command(label="Flip", command=do_flip)
    preprocessmenu.add_command(label="Scale", command=do_scale)
    menubar.add_cascade(label="Preprocess", menu=preprocessmenu)

    root.config(menu=menubar)

    current_mesh_label = Label(root, text="Current mesh")
    current_mesh_label.grid(row=0, column=0)
    current_csv_label = Label(root, text="Current CSV table")
    current_csv_label.grid(row=0, column=1)

    label_loaded_meshes = Label(root, text="Loaded meshes")
    label_loaded_meshes.grid(row=1, column=0)
    listbox_loaded_meshes = Listbox(root, width=50)
    listbox_loaded_meshes.grid(row=2, column=0, columnspan=3)
    #button_graph = Button(root, text="Show histogram", command=draw_histogram(selected_x, selected_y))
    #button_graph.grid(row=3, column=1)
    def callback(event):
        global data
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            data = event.widget.get(index)
            current_mesh_label.configure(text=f"Current mesh:{data}")

    listbox_loaded_meshes.bind("<<ListboxSelect>>", callback)
    root.mainloop()



if __name__ == "__main__":
    main()