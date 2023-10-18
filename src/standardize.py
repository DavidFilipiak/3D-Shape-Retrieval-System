import numpy as np



def standardize_scalar_features(df, features):
    new_df = df.copy()
    for f in features:
        new_df[f] = (new_df[f] - new_df[f].mean()) / new_df[f].std()
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