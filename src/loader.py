import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


def read_features(filename='data/features.txt'):
    with open(filename, 'r') as f:
        table = pd.read_csv(f, delim_whitespace=True, header=None)
    return table


def read_labels(filename='data/labels.txt'):
    with open(filename, 'r') as f:
        table = pd.read_csv(f, squeeze=True, header=None, names=["Category"])
    return table


def dump_pickle(feature_path='data/features.txt',
                label_path='data/labels.txt',
                output_path='data/dataset.pik'):
    "dump the dataset into pickles, the dumped files will be used by get_dataset()"
    features = read_features(feature_path)
    labels = read_labels(label_path).astype("category")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=0.4, random_state=42)
    with open(output_path, 'wb') as f:
        pickle.dump({"X_train": X_train,
                     "X_test": X_test,
                     "y_train": y_train,
                     "y_test": y_test}, f)


def get_dataset(data_path='data/dataset.pik'):
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset["X_train"].to_numpy(), dataset["X_test"].to_numpy(), dataset["y_train"].to_numpy(), dataset[
        "y_test"].to_numpy()


if __name__ == '__main__':
    dump_pickle()
