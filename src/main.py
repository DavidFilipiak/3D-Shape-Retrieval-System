from tkinter import *
from tkinter import filedialog
import os
import pymeshlab
import polyscope as ps
from mesh import Mesh

# GLOBAL VARIABLES
ms = None
listbox_loaded_meshes = None

current_dir = os.getcwd()
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
def browse_button() -> None:
  db_dir = os.path.abspath(os.path.join(current_dir, "..", "db"))
  filename = filedialog.askopenfilename(title="Mesh select", initialdir=db_dir, filetypes=[('Mesh files', '*.obj')])
  ms.load_new_mesh(filename)
  classType = os.path.dirname(filename).split('/')[-1]

  print(f"class type: {classType}")
  print(f"face number: {ms.current_mesh().face_number()}")
  print(f"The mesh is alligned on the x-axis: {ms.current_mesh().bounding_box().dim_x()}, y-axis: {ms.current_mesh().bounding_box().dim_y()}"
        f" and z-axis: {ms.current_mesh().bounding_box().dim_z()}")
  print(f" vertex number{ms.current_mesh().vertex_number()}")
  listbox_loaded_meshes.insert(END, "/".join(filename.split("/")[-2:]))
  num_triangles, num_quads = count_triangles_and_quads(ms.current_mesh().polygonal_face_list())
  print(f"The number of triangles is {num_triangles}, and the number of quads is: {num_quads}")


def main() -> None:
  global ms, listbox_loaded_meshes
  ms = pymeshlab.MeshSet()

  root = Tk()
  root.title("3D Shape Retrieval")
  root.geometry("500x500")

  button_browse = Button(text="Load Mesh", command=browse_button)
  button_browse.grid(row=0, column=1)
  button_show = Button(text="Show Loaded Meshes", command=ms.show_polyscope)
  button_show.grid(row=0, column=2)

  label_loaded_meshes = Label(root, text="Loaded Meshes")
  label_loaded_meshes.grid(row=1, column=1)
  listbox_loaded_meshes = Listbox(root, width=50)
  listbox_loaded_meshes.grid(row=2, column=1, columnspan=3)


  # class_type_label = Label(root, text="Class type: N/A")
  # class_type_label.grid(row=1, column=1)
  #
  # face_number_label = Label(root, text="Face number: N/A")
  # face_number_label.grid(row=2, column=1)
  #
  # vertex_number_label = Label(root, text="Vertex number: N/A")
  # vertex_number_label.grid(row=3, column=1)



  root.mainloop()

if __name__ == "__main__":
  main()