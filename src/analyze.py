from __future__ import annotations
import pandas as pd
import numpy as np
import math
from sklearn.manifold import TSNE
from utils import flatten_nested_array
from feature import descriptor_shape_list, descriptor_list


# basically an interface
class Analysis:
    min_view = 0
    max_view = 0
    mean_view = 0
    mean_view_mesh = ""  # the mesh closest to the mean from the whole dataset
    mean_view_mesh_value = 0
    std_view = 0

    avg_mesh_value = 0  # value of the mesh closest to the mean
    avg_mesh = ""
    min_value = 0
    min_mesh = ""
    max_value = 0
    max_mesh = ""

    mean_all = 0
    std_all = 0
    outliers = np.asarray([])
    histogram = (np.asarray([]), np.asarray([]))
    values = np.asarray([])
    all = np.asarray([])

    x_axis_label = "Bin size"
    y_axis_label = "Number of meshes"

    def __init__(self):
        pass


def analyze_feature_all(table: pd.DataFrame, min_view: float, max_view: float) -> Analysis:
    table = pd.DataFrame(table)
    if len(table.columns) != 2:
        raise Exception("Table must have 2 columns. The 'name' column and the feature column to be analyzed")

    feature = [col for col in table.columns if col != "name"][0]
    table.sort_values(feature, inplace=True)

    array = table[feature].values
    outliers = array[np.where((array > max_view) | (array < min_view))]
    values = array[np.where((array <= max_view) & (array >= min_view))]

    mean = np.mean(values)
    std = np.std(values)
    max_value = table[feature].max()
    row_with_max_value = table[table[feature] == max_value]
    min_value = table[feature].min()
    row_with_min_value = table[table[feature] == min_value]
    average_value = table[feature].mean()
    std_all = table[feature].std()
    table['temp_abs_diff'] = abs(table[feature] - average_value)
    closest_avg_row = table[table['temp_abs_diff'] == table['temp_abs_diff'].min()]
    table['temp_abs_diff'] = abs(table[feature] - mean)
    closest_mean_row = table[table['temp_abs_diff'] == table['temp_abs_diff'].min()]
    closest_avg_row = closest_avg_row.drop('temp_abs_diff', axis=1)[["name", feature]]
    closest_mean_row = closest_mean_row.drop('temp_abs_diff', axis=1)[["name", feature]]
    table.drop('temp_abs_diff', axis=1, inplace=True)

    histogram = np.histogram(values, bins=math.ceil(math.sqrt(len(values))))

    analysis = Analysis()
    analysis.min_view = min_view
    analysis.max_view = max_view
    analysis.min_value = min_value
    analysis.min_mesh = row_with_min_value["name"].values[0]
    analysis.max_value = max_value
    analysis.max_mesh = row_with_max_value["name"].values[0]
    analysis.avg_mesh = closest_avg_row["name"].values[0]
    analysis.avg_mesh_value = closest_avg_row[feature].values[0]
    analysis.mean_all = average_value
    analysis.mean_view_mesh = closest_mean_row["name"].values[0]
    analysis.mean_view_mesh_value = closest_mean_row[feature].values[0]
    analysis.std_all = std_all
    analysis.mean_view = mean
    analysis.std_view = std
    analysis.outliers = outliers
    analysis.histogram = histogram
    analysis.values = values
    analysis.all = array
    analysis.y_axis_label = table.columns[1]
    analysis.x_axis_label = "Meshes"
    return analysis


def analyze_bary_distance_to_origin_all(table: pd.DataFrame, colName: str):
    extract = pd.DataFrame(table["name"])
    array = np.asarray(table[colName].tolist())
    distances = np.linalg.norm(array, axis=1)
    extract["distances"] = pd.Series(distances).values

    return analyze_feature_all(extract, 0, 20)


def analyze_major_eigenvector_dot_with_x_axis(table: pd.DataFrame, colName: str):
    extract = pd.DataFrame(table["name"])
    array = np.asarray(table[colName].tolist())
    x_axis = np.asarray([1, 0, 0])
    dot_products = np.absolute(np.dot(array, x_axis))
    extract["dot_products"] = pd.Series(dot_products).values

    return analyze_feature_all(extract, 0, 1)

def analyze_median_eigenvector_dot_with_y_axis(table: pd.DataFrame, colName: str):
    extract = pd.DataFrame(table["name"])
    array = np.asarray(table[colName].tolist())
    y_axis = np.asarray([0, 1, 0])
    dot_products = np.absolute(np.dot(array, y_axis))
    extract["dot_products"] = pd.Series(dot_products).values

    return analyze_feature_all(extract, 0, 1)

def analyze_minor_eigenvector_dot_with_z_axis(table: pd.DataFrame, colName: str):
    extract = pd.DataFrame(table["name"])
    array = np.asarray(table[colName].tolist())
    z_axis = np.asarray([0, 0, 1])
    dot_products = np.absolute(np.dot(array, z_axis))
    extract["dot_products"] = pd.Series(dot_products).values

    return analyze_feature_all(extract, 0, 1)

def analyze_mass_orientation(table: pd.DataFrame, colName: str):
    extract = pd.DataFrame(table["name"])
    array = np.asarray(table[colName].tolist())
    arr_x = array[:, 0]
    arr_y = array[:, 1]
    arr_z = array[:, 2]

    extract["mass_x"] = pd.Series(arr_x).values
    analysis_x = analyze_feature_all(extract, -1, 1)
    extract = extract.drop("mass_x", axis=1)
    extract["mass_y"] = pd.Series(arr_y).values
    analysis_y = analyze_feature_all(extract, -1, 1)
    extract = extract.drop("mass_y", axis=1)
    extract["mass_z"] = pd.Series(arr_z).values
    analysis_z = analyze_feature_all(extract, -1, 1)

    return (analysis_x, analysis_y, analysis_z)


def reduce_tsne(df):
    df = df.copy()
    df_values = df.drop(columns=['name', 'class_name'])
    X = [[] for _ in range(len(df_values.values))]
    desc_columns = df_values.columns

    for i, elem_descriptor in enumerate(descriptor_list):
        if elem_descriptor in desc_columns:
            for j, row in enumerate(df_values[elem_descriptor].values):
                X[j].append(row[1])
    for hist_descriptor in descriptor_shape_list:
        if hist_descriptor in desc_columns:
            df_values[hist_descriptor] = df_values[hist_descriptor].apply(lambda x: x[1])
            for j, row in enumerate(df_values[hist_descriptor].values):
                X[j].extend(list(row))

    values = np.asarray(X, dtype=np.float64)
    values_tsne = TSNE(n_components=2, perplexity=10).fit_transform(values)
    df_tsne = pd.DataFrame(values_tsne, columns=['x', 'y'])
    df_tsne.insert(0, 'name', df['name'])
    df_tsne.insert(1, 'class_name', df['class_name'])
    return df_tsne





