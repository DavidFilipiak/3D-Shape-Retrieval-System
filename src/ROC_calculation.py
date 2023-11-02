import os
import pickle
import numpy as np
import pandas as pd
from src.analyze import reduce_tsne_from_dist_matrix, reduce_tsne
from src.database import Database
from sklearn.neighbors import KDTree
import json


def main():
    database = Database()
    current_dir = os.getcwd()
    filename = os.path.abspath(os.path.join(current_dir, "csv_files", "Repaired_meshes_final_standardized.csv"))
    database.load_table(filename)
    df1 = database.get_table()
    filename = os.path.abspath(
        os.path.join(current_dir, "csv_files", "shape_descriptors_for_querying_standardized.csv"))
    database.clear_table()
    database.load_table(filename)
    df2 = database.get_table()
    result = pd.merge(df1, df2, on=['name', 'class_name'], how='inner')
    result = result.sort_values(by=['name'], ignore_index=True)

    class_counts = result['name'].str.split('/').str[0].value_counts().to_dict()

    # Metrics with volume (saved kd-tree)
    df_tsne = reduce_tsne(result)
    values = df_tsne.iloc[:, 2:].to_numpy()
    tree = KDTree(values)
    #with open('tree.pickle', 'rb') as file_handle:
    #    tree = pickle.load(file_handle)

    # Metrics with volume (saved distance matrix) Don't delete PLS :3
    #distance_matrix = np.load("dist_matrix.npy")
    #shape2idx = json.load(open("shape2idx.json", "r"))
    #bad_quadruped_index = shape2idx["Quadruped/m94.obj"]
    #distance_matrix = np.delete(distance_matrix, bad_quadruped_index, 0)
    #distance_matrix = np.delete(distance_matrix, bad_quadruped_index, 1)
    #df_tsne = reduce_tsne_from_dist_matrix(distance_matrix)
    #values = df_tsne.iloc[:, 2:].to_numpy()
    #tree = KDTree(values)

    # Metrics
    metrics_dict = {
        "TP": 0,
        "FN": 1,
        "FP": 2,
        "TN": 3,
        "Accuracy": 4,
        "Precision": 5,
        "Recall": 6,
        "F1": 7,
        "Sensitivity": 8,
        "Specificity": 9
    }
    query_sizes = range(1, 101, 5)  # Assuming these are the sizes we are interested in
    query_size_mapper = {val: i for i, val in enumerate(query_sizes)}
    # Initialize the 3D array for storing metrics
    num_shapes = result.shape[0]
    num_sizes = len(query_sizes)
    num_metrics = len(metrics_dict)
    metrics_array = np.zeros((num_shapes, num_sizes, num_metrics))

    for i in query_sizes:
        for _, row in result.iterrows():
            query_shape = row['name']
            print(i, query_shape)
            query_class = query_shape.split('/')[0]
            class_size = class_counts[query_class]
            # retrieved_shapes
            distances, indices = tree.query(values[row.name:row.name + 1], k=i)
            retrieved_shapes = result.iloc[indices[0]]['name'].values
            retrieved_classes = [shape.split('/')[0] for shape in retrieved_shapes]
            metrics = calculate_metrics(result, retrieved_classes, query_class, class_size)
            metrics_array[row.name, query_size_mapper[i], :] = metrics

    np.save("metrics.npy", metrics_array)
    print("FINISHED")

# Function to calculate metrics
def calculate_metrics(result, retrieved_classes, query_class, query_class_size=0):
    TP = retrieved_classes.count(query_class)
    # Calculate FP
    FP = len(retrieved_classes) - TP
    # Calculate FN
    FN = query_class_size - TP
    # Calculate TN
    TN = result.shape[0] - query_class_size - FP
    Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0
    Sensitivity = Recall
    Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return [TP, FN, FP, TN, Accuracy, Precision, Recall, F1, Sensitivity, Specificity]


if __name__ == '__main__':
    main()




