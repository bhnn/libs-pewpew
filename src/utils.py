import numpy as np

# transform full mineral label of 0-108 to 0-11 for reduced dataset
reduced_labels_classes = {
    1: 0,
    3: 1,
    4: 2,
    6: 3,
}
reduced_labels_subgroups = {
    1:  0,
    4:  1,
    5:  2,
    9:  3,
    18: 4,
}
reduced_labels_minerals = {
    11:  0,
    19:  1,
    26:  2,
    28:  3,
    35:  4,
    41:  5,
    73:  6,
    80:  7,
    86:  8,
    88:  9,
    97: 10,
    98: 11,
}

def normalise_minmax(np_data):
    result = np.zeros((np_data.shape[0], np_data.shape[1]))

    for i in range(np_data.shape[0]):
        if np.max(np_data[i,:]) > 0:
            result[i] = np_data[i][:,1] / np.max(np_data[i][:,1])
        else:
            raise ValueError
    return result

def transform_labels(label_data, cell=2):
    if cell == 0:
        for i in range(len(label_data)):
            label_data[i][cell] = reduced_labels_classes[label_data[i][cell]]
    elif cell == 1:
        for i in range(len(label_data)):
            label_data[i][cell] = reduced_labels_subgroups[label_data[i][cell]]
    elif cell == 2:
        for i in range(len(label_data)):
            label_data[i][cell] = reduced_labels_minerals[label_data[i][cell]]
    return label_data