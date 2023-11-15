# 3D-Shape-Retrieval-System

This project is an implementation of a simple 3D shape retrieval system.
It allows to load 3D meshes, preprocess them, extract features, compute distances between meshes, and query the database for similar shapes.

### For using the application through the terminal
* step 1) select the main branch from the repository
* step 2) download the project folder
* step 3) change path to the project folder in the command line by using "cd /user path"
* step 4) execute the command on the terminal "pip3 install -r requirements.txt"
* step 5) from command line run "python main.py"

alternatively, you can run the application from the IDE by running the main.py file.

## UI Instructions
### File menu
* __Load Mesh (.obj)__: opens a file dialog to select a (.obj) file to load
* __Load All (.obj)__: opens a file dialog to select a folder containing (.obj) files to load
* __Load All (.csv)__: opens a file dialog to select a precomputed (.csv) file to load
* __Clear All/Selected__: deletes the respective meshes from the application
* __Save Current Mesh/ALl (.obj)__: opens a file dialog to select a folder to save the current mesh or all meshes in the application
* __Save ... (.csv)__: opens a file dialog to select a folder to save the computed features of all loaded meshes

### Query menu
* __Naive Feature Weighting__: opens a file dialog and returns 5 nearest shapes to the selected shape using naive search method on weighted features
* __Naive Distance Weighting__: opens a file dialog and returns 5 nearest shapes to the selected shape using naive search method on weighted distances using a custom distance function
* __kD-tree (no DR)__: opens a file dialog and returns 5 nearest shapes to the selected shape using kD-tree search method on weighted distances using a custom distance function
* __kD-tree (t-SNE)__: opens a file dialog and returns 5 nearest shapes to the selected shape using kD-tree search method build from a t-SNE reduction and queried on weighted distances using a custom distance function

### Show menu
* __Show all loaded meshes__: opens a new window and displays all loaded meshes

### Analyze menu
* __Current mesh__: displays the analysis of a specified feature for the currently selected mesh
* __All features__: displays the analysis of a specified feature for all loaded meshes (requires to load the corresponding (.csv) file)
* __All descriptors__: displays the analysis of a specified descriptor for all loaded meshes (requires to load the corresponding (.csv) file)
* __Dimensionality Reduction__: displays the analysis of a specified dimensionality reduction for the entire dataset.
* __Distance matrix__: displays a heatmap of a distance matrix computed by the custom distance function

### Preprocess menu
* __Batch__: opens a file dialog to select a folder containing (.obj) files to preprocess and saves them as new (.obj) files
* __Full__: does the full preprocessing pipeline on the currently loaded meshes
* __Translate__, ..., __Align__: performs the respective preprocessing step on the currently loaded meshes

### Extract menu
* __Extract features__: extracts elementary features from all currently selected meshes
* __Batch Shape descriptors__: opens a file dialog to select a folder containing (.obj) files to extract shape descriptors from, and saves them in a (.csv) file
* __A3__, ..., __D4__: extracts the respective shape descriptor from all currently selected meshes

### Standardize menu
* __Standardize elementary descriptors__: standardizes the values of extracted feature descriptors loaded from a respective (.csv) file
* __Standardize histogram shape descriptors__: standardizes the values of extracted histogram shape descriptors loaded from a respective (.csv) file

### Evaluate menu
* __Evaluation__: creates a new (.csv) file containing the 5 nearest neighbors of all loaded meshes using a kD-tree (t-SNE) search method
* __Compute metrics__: computes the TP, FP, TN and FN for the currently loaded meshes using a kD-tree (t-SNE) search method