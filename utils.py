import csv
import numpy as np


# read csv file into list
def csv_reader(csvfile):
    list_data = []
    with open(csvfile) as f:
        reader = csv.reader(f)
        for row in reader:
            list_data.append(row)
    # remove first row and first col
    label = list_data[0]
    list_data = list_data[1:]
    for i in range(len(list_data)):
        list_data[i] = list_data[i][1:]
    return list_data, label


# if a 1D list is composed of all digits
def is_digit(list_data):
    flag = True
    for x in list_data:
        if x.isdigit():
            continue
        else:
            flag = False
    return flag


# delete specified column in a 2D list
def remove_col(list_data, col_num):
    data = list_data
    for i in range(len(data)):
        del data[i][col_num]
    return data


# extract specified column into 1D list from a 2D list
def extract_column(list_data, col_num):
    data_col = []
    for i in range(len(list_data)):
        data_col.append(list_data[i][col_num])
    return data_col


# if a list contains sufficient different values
def not_diverse(col_data):
    number_of_values = 0
    data = []
    for x in col_data:
        if x not in data:
            number_of_values += 1
            data.append(x)
    if number_of_values < len(col_data) / 10:
        return True
    else:
        return False


# get the data of both contains only digits and diverse enough
def get_data_col_index(list_data):
    col_idx = np.array([100])
    for i in range(len(list_data[0])):
        col = extract_column(list_data, i)
        if is_digit(col) and not not_diverse(col):
            tmp = np.array([i])
            col_idx = np.concatenate((col_idx, tmp))
    col_idx = col_idx[1:]
    return col_idx


# def process_data(list_data):
#     i = 0
#     while i < len(list_data[0]):
#         col = extract_column(list_data, i)
#         if not is_digit(col):
#             list_data = remove_col(list_data, i)
#         elif not_diverse(col):
#             list_data = remove_col(list_data, i)
#         else:
#             i += 1
#     return list_data


# convert digit list into numpy array
def list_to_array(list_data):
    data = np.zeros((len(list_data), len(list_data[0])))
    for i in range(len(list_data)):
        for j in range(len(list_data[i])):
            data[i][j] = list_data[i][j]
    return data


# get multiple columns from a 2D list
def get_specified_col_list(list_data, col_index):
    data = []
    for i in range(len(list_data)):
        tmp = []
        for j in range(col_index.size):
            tmp.append(list_data[i][col_index[j]])
        data.append(tmp)
    return data
