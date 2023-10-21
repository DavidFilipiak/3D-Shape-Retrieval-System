import numpy as np
from matplotlib import pyplot as plt
from utils import draw_histogram


def standardize_scalar_features(df, features):
    new_df = df.copy()
    for f in features:
        values = new_df[f].values
        hist = np.histogram(values, bins=100)
        print(max(values))
        draw_histogram(hist[1][:-1], hist[0], min(values), max(values), ylabel=str(f)+" before")
        new_df[f] = (new_df[f] - new_df[f].mean()) / new_df[f].std()
        values = new_df[f].values
        hist = np.histogram(values, bins=100)
        print(max(values))
        draw_histogram(hist[1][:-1], hist[0], min(values), max(values), ylabel=str(f)+" after")
    return new_df


def standardize_histogram_features(df, features):
    new_df = df.copy()
    for f in features:
        new_df[f] = new_df[f].apply(normalize_histogram)
    return new_df

def normalize_histogram(histogram):
    hist_y = histogram[1]
    hist_y_norm = hist_y / hist_y.sum()
    return np.array([histogram[0], hist_y_norm])