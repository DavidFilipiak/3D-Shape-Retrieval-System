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

def browse_button() -> None:
  db_dir = os.path.abspath(os.path.join(current_dir, "..", "db"))
  filename = filedialog.askopenfilename(title="Mesh select", initialdir=db_dir, filetypes=[('Mesh files', '*.obj')])
  ms.load_new_mesh(filename)
  listbox_loaded_meshes.insert(END, "/".join(filename.split("/")[-2:]))
  #ms.show_polyscope()


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

  root.mainloop()

if __name__ == "__main__":
  main()