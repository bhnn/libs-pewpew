import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Input

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

def prepare_dataset(dataset_choice, targets, num_classes, batch_size=64):
    if dataset_choice == 0:
        train_samples, train_labels, test_samples, test_labels = np.load('/home/ben/Desktop/ML/synthetic_data/final.npy', allow_pickle=True)
        data_str = 'synthetic data'
    elif dataset_choice == 1:
        train_samples, train_labels, test_samples, test_labels = np.load('/home/ben/Desktop/ML/pretty_data/final.npy', allow_pickle=True)
        data_str = 'handheld data'
    elif dataset_choice == 2:
        train_samples_1, train_labels_1, test_samples_1, test_labels_1 = np.load('/home/ben/Desktop/ML/synthetic_data/final.npy', allow_pickle=True)
        train_samples_2, train_labels_2, test_samples_2, test_labels_2 = np.load('/home/ben/Desktop/ML/pretty_data/final.npy', allow_pickle=True)
        train_samples = np.vstack((train_samples_1, train_samples_2))
        test_samples = np.vstack((test_samples_1, test_samples_2))
        train_labels = np.vstack((train_labels_1, train_labels_2))
        test_labels = np.vstack((test_labels_1, test_labels_2))
        data_str = 'both synthetic and handheld data'
    else:
        raise ValueError('Invalid dataset parameter')

    # normalise each sample with its own np.max
    train_samples_norm = normalise_minmax(train_samples)
    test_samples_norm = normalise_minmax(test_samples)
    # create onehot vectors out of labels
    train_labels = transform_labels(train_labels, cell=targets)
    test_labels = transform_labels(test_labels, cell=targets)
    train_labels = train_labels[:,targets].astype(int)
    test_labels = test_labels[:,targets].astype(int)
    train_labels_onehot = to_categorical(train_labels, num_classes)
    test_labels_onehot = to_categorical(test_labels, num_classes)
    # use Dataset as input pipeline
    train_data = tf.data.Dataset.from_tensor_slices((train_samples_norm, train_labels_onehot)).shuffle(10000).batch(batch_size, drop_remainder=True).repeat(-1)
    test_data  = tf.data.Dataset.from_tensor_slices((test_samples_norm, test_labels_onehot)).batch(batch_size)

    epoch_steps = train_samples.shape[0] // batch_size

    return train_data, test_data, train_labels, test_labels, epoch_steps, data_str

def set_classification_targets(cls_choice):
    if cls_choice == 0:
        num_classes = 4
        cls_target= 0
        cls_str = 'mineral classes'
    elif cls_choice == 1:
        num_classes = 5
        cls_target = 1
        cls_str = 'mineral subgroups'
    elif cls_choice == 2:
        num_classes = 12
        cls_target = 2
        cls_str = 'minerals'
    else:
        raise ValueError('Invalid classification target parameter')
    return num_classes, cls_target, cls_str

def build_model(id, num_classes, name='model', inputs=None, new_input=False, reg=regularizers.l2, reg_lambda=0.0001):
    model_name = name if name else f'model_{id}'
    with tf.name_scope(model_name):
        if new_input:
            inputs = Input(shape=(7810,))
        net = Dense(512, activation='relu', kernel_regularizer=reg(reg_lambda))(inputs)
        net = Dropout(0.5)(net)
        net = Dense(256, activation='relu', kernel_regularizer=reg(reg_lambda))(net)
        net = Dropout(0.5)(net)
        net = Dense(128, activation='relu', kernel_regularizer=reg(reg_lambda))(net)
        net = Dropout(0.5)(net)
        net = Dense(64, activation='relu', kernel_regularizer=reg(reg_lambda))(net)
        net = Dropout(0.5)(net)
        net = Dense(num_classes, activation='softmax')(net)

    model = keras.Model(inputs=inputs, outputs=net, name=name)
    return model

def build_model_concat(id, num_classes, inputs=None, new_input=False, concat_model=None, reg=regularizers.l2, reg_lambda=0.0001):
    model_name = name if name else f'model_{id}'
    with tf.name_scope(model_name):
        if new_input:
            inputs = Input(shape=(7810,))
        if concat_model:
            inputs = Concatenate()([concat_model.inputs[0], concat_model.layers[-2].output])
        net = Dense(512, activation='relu', kernel_regularizer=reg(reg_lambda))(inputs)
        net = Dropout(0.5)(net)
        net = Dense(256, activation='relu', kernel_regularizer=reg(reg_lambda))(net)
        net = Dropout(0.5)(net)
        net = Dense(128, activation='relu', kernel_regularizer=reg(reg_lambda))(net)
        net = Dropout(0.5)(net)
        net = Dense(64, activation='relu', kernel_regularizer=reg(reg_lambda))(net)
        net = Dropout(0.5)(net)
        if not concat_model:
            net = Dense(7810, activation='relu', kernel_regularizer=reg(reg_lambda))(net)
            net = Dropout(0.5)(net)
        net = Dense(num_classes, activation='softmax')(net)

    if not concat_model:
        model = keras.Model(inputs=inputs, outputs=net)
    else:
        model = keras.Model(inputs=concat_model.inputs[0], outputs=net)
    return model