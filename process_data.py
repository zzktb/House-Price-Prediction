from utils import csv_reader, list_to_array, get_data_col_index, extract_column, get_specified_col_list
import csv
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize


def process_data(csv_train_file, num_of_train_data):
    data_raw, lb = csv_reader(csv_train_file)
    col_index = get_data_col_index(data_raw)
    print("Feature index: \n\t", col_index)
    label = []
    for i in col_index:
        label.append(lb[i + 1])
    print("Feature name: \n\t", label)
    data = get_specified_col_list(data_raw, col_index)
    data = list_to_array(data)
    data_s = shuffle(data, random_state=13)
    offs = num_of_train_data

    X_train = data_s[:offs, :-1]
    X_train = normalize(X_train, axis=0)
    X_test = data_s[offs:, :-1]
    X_test = normalize(X_test, axis=0)
    y_train = data_s[:offs, -1]
    y_train = y_train / np.linalg.norm(y_train)
    y_test = data_s[offs:, -1]
    y_test_norm = np.linalg.norm(y_test)
    y_test = y_test / y_test_norm
    return X_train, y_train, X_test, y_test, y_test_norm, label

