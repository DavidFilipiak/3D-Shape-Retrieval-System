from tkinter import *
from tkinter import filedialog
import os
import pymeshlab
import polyscope as ps
from mesh import Mesh
ms = None

current_dir = os.getcwd()

def browse_button() -> None:
  db_dir = os.path.abspath(os.path.join(current_dir, "..", "db"))
  filename = filedialog.askopenfilename(title="Mesh select", initialdir=db_dir, filetypes=[('Mesh files', '*.obj')])
  ms.load_new_mesh(filename)
  ms.show_polyscope()


def main() -> None:
  global ms
  ms = pymeshlab.MeshSet()

  root = Tk()
  root.title("3D Shape Retrieval")
  root.geometry("500x500")

  label_browse = Label(root, text="Load Mesh into the app ")
  label_browse.grid(row=0, column=1)
  button_browse = Button(text="Load", command=browse_button)
  button_browse.grid(row=0, column=3)

  root.mainloop()

if __name__ == "__main__":
  main()