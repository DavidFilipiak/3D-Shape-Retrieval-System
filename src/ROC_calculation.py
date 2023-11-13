import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
    filtered_df = result[result['class_name'].isin(['Sign', 'Bicycle'])]
    result = filtered_df
    class_counts = result['name'].str.split('/').str[0].value_counts().to_dict()

    # Metrics with volume (saved kd-tree)
    #df_tsne = reduce_tsne(result)
    #values = df_tsne.iloc[:, 2:].to_numpy()
    #tree = KDTree(values)
    #with open('tree.pickle', 'rb') as file_handle:
    #    tree = pickle.load(file_handle)

    # Metrics with volume (saved distance matrix) Don't delete PLS :3
    distance_matrix = np.load("dist_matrix.npy")
    shape2idx = json.load(open("shape2idx.json", "r"))
    bad_quadruped_index = shape2idx["Quadruped/m94.obj"]
    distance_matrix = np.delete(distance_matrix, bad_quadruped_index, 0)
    distance_matrix = np.delete(distance_matrix, bad_quadruped_index, 1)
    df_tsne = reduce_tsne_from_dist_matrix(distance_matrix)
    values = df_tsne.iloc[:, 2:].to_numpy()
    tree = KDTree(values)

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
    query_sizes = range(1, result.shape[0], 1)  # Assuming these are the sizes we are interested in
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

    np.save("metrics-all.npy", metrics_array)

    #metrics_array = np.load("metrics-best.npy")
    plot_average_sensitivity_specificity(metrics_array, metrics_dict, num_shapes, num_sizes, result)
    plot_overall_average_sensitivity_specificity(metrics_array, metrics_dict, num_sizes)

    print("FINISHED")


def plot_average_sensitivity_specificity(metrics_array, metrics_dict, num_shapes, num_sizes, result):
    # Get a list of unique classes
    classes = result['name'].str.split('/').str[0].unique()

    # Iterate over each class to plot their average sensitivity and specificity
    for class_name in classes:
        avg_sens = np.zeros(num_sizes)  # Array to store average Sensitivity values
        avg_specs = np.zeros(num_sizes)  # Array to store average Specificity values
        class_shape_indices = result[result['name'].str.contains(f'^{class_name}/')].index

        # Calculate the average sensitivity and specificity for each query size
        for size_idx in range(num_sizes):
            sens_values = []
            specs_values = []
            for shape_idx in class_shape_indices:
                sens_values.append(metrics_array[shape_idx, size_idx, metrics_dict["Sensitivity"]])
                specs_values.append(metrics_array[shape_idx, size_idx, metrics_dict["Specificity"]])
            # Compute the average for the current query size
            avg_sens[size_idx] = np.mean(sens_values)
            avg_specs[size_idx] = np.mean(specs_values)

        # Plot the average Sensitivity vs Specificity curve for this class
        plt.plot(avg_sens, avg_specs, label=f'Class: {class_name}')

    # Add labels and title to the plot
    plt.xlabel('Average Sensitivity')
    plt.ylabel('Average Specificity')
    plt.title('Average Sensitivity vs Specificity by Class')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def plot_overall_average_sensitivity_specificity(metrics_array, metrics_dict, num_sizes):
    avg_sens = np.zeros(num_sizes)  # Array to store overall average Sensitivity values
    avg_specs = np.zeros(num_sizes)  # Array to store overall average Specificity values

    # Calculate the average sensitivity and specificity for each query size across all shapes
    for size_idx in range(num_sizes):
        sens_values = metrics_array[:, size_idx, metrics_dict["Sensitivity"]]
        specs_values = metrics_array[:, size_idx, metrics_dict["Specificity"]]

        # Compute the average for the current query size across all shapes
        avg_sens[size_idx] = np.mean(sens_values)
        avg_specs[size_idx] = np.mean(specs_values)

    # Plot the overall average Sensitivity vs Specificity curve
    plt.plot(avg_sens, avg_specs, label='Overall Average')

    # Add labels, title, and legend to the plot
    plt.xlabel('Average Sensitivity')
    plt.ylabel('Average Specificity')
    plt.title('Overall Average Sensitivity vs Specificity')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()

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

def custom_eval():
    metrics = np.load("metrics_more_classes.npy")
    print(metrics.shape)
    metrics_dict = {"TP": 0,"FN": 1,"FP": 2,"TN": 3,"Accuracy": 4,"Precision": 5,"Recall": 6,"F1": 7,"Sensitivity": 8,"Specificity": 9}
    query_sizes = range(1, 2477, 1)  # Assuming these are the sizes we are interested in
    query_size_mapper = {val: i for i, val in enumerate(query_sizes)}
    old_shape2idx = json.load(open("shape2idx.json", "r"))
    bad_quadruped_index = old_shape2idx["Quadruped/m94.obj"]
    shape2idx = dict()
    for i, (name, _) in enumerate(old_shape2idx.items()):
        if i < bad_quadruped_index:
            shape2idx[name] = i
        elif i > bad_quadruped_index:
            shape2idx[name] = i - 1
    classes = sorted(list(set([name.split("/")[0] for name in shape2idx.keys()])))

    # Plot custom histograms - TWO out of THREE following variables must be filled in, the third one can must be None
    # None in this case means ALL values of that variable
    class_name = None
    query_size = 5
    metric = "Accuracy"

    if not class_name:
        class_indexes = list(range(0, 2477))
    else:
        class_indexes = \
        list(
            map(lambda item: item[1],
                filter(
                    lambda item: item[0].split("/")[0] == class_name,
                    shape2idx.items()
                )
            )
        )

    if not query_size:
        query_size_indexes = list(query_size_mapper.values())
    else:
        query_size_indexes = [query_size_mapper[query_size]]

    if not metric:
        metric_indexes = list(metrics_dict.values())
    else:
        metric_indexes = [metrics_dict[metric]]

    matrix_window = metrics[
                        min(class_indexes):max(class_indexes)+1,
                        min(query_size_indexes):max(query_size_indexes)+1,
                        min(metric_indexes):max(metric_indexes)+1]

    matrix_window = np.squeeze(matrix_window)
    if class_name:
        matrix_window = np.mean(matrix_window, axis=0)
    else:
        df = pd.DataFrame({"class_name": [key.split("/")[0] for key in shape2idx.keys()], "value": list(matrix_window)})
        df = df.groupby("class_name").mean()
        matrix_window = df["value"].to_numpy()


    if not class_name:
        #plt.figure(figsize=(10, 8))
        plt.title(f"histogram of {metric} with query size {query_size} across all classes")
        reordered_classes = list(sorted(classes, key=lambda x: matrix_window[classes.index(x)]))
        plt.bar([i for i in range(len(matrix_window))], list(sorted(matrix_window)))
        plt.xticks([i for i in range(len(classes))], reordered_classes, rotation=90)
    elif not query_size:
        plt.title(f"Histogram of class {class_name}")
        plt.bar([i for i in range(len(matrix_window))], matrix_window)
        #plt.xticks([query_size_mapper[i] for i in query_sizes] ,query_sizes, rotation=50)
    elif not metric:
        plt.title(f"Histogram of query size {query_size}")
        plt.bar([i for i in range(len(matrix_window))][4:], matrix_window[4:])
        plt.xticks([i for i in range(len(metrics_dict))][4:], list(metrics_dict.keys())[4:], rotation=50)

    plt.show()




if __name__ == '__main__':
    main()
    #custom_eval()




