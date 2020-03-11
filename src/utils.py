from glob import glob

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras import Model, regularizers
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

# adapted from Federico
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

def has_sufficient_copper(spectrum, amount_t=0.5, drop_t=0.3):
    """
    Checks individual spectras for their copper content. If more than *amount_t*% of copper lines are below the *drop_t*
    percentile, then the spectrum more likely captures matrix rock.
    Parameters:
        spectrum (ndarray):     numpy array containing the spectrum
        amount_t (float):       threshold (%) of copper lines expected to have high intensity
        drop_t   (float):       threshold (%) signifying "low" copper intensity
    
    Returns:
        *True* if the spectrum contains more than the specified amount of copper, *False* otherwise
    """
    # lookup for position of wavelength in input data
    copper_lines_indices = [int(round(cpl - 180, 1) * 10) for cpl in [219.25, 224.26, 224.7, 324.75, 327.4, 465.1124, 510.55, 515.32, 521.82]]
    low_copper_sum = 0
    for cli in copper_lines_indices:
        # find position in data from wavelength. 219.25 -> 219.2 -> 39.2 -> data[392] == [219.2, ...]
        if spectrum[cli] < drop_t:
            low_copper_sum += 1
        if (low_copper_sum / len(copper_lines_indices)) > amount_t:
            return False
    return True

def prepare_dataset(dataset_choice, targets, num_classes, batch_size=64, return_tf_dataset=True, amount_t=None, drop_t=None):
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

    # check for copper content if possible, discard any elements that have too little copper
    if amount_t and drop_t:
        train_logical_indices = [has_sufficient_copper(s, amount_t, drop_t) for s in train_samples_norm]
        test_logical_indices = [has_sufficient_copper(s, amount_t, drop_t) for s in test_samples_norm]
        # discard by using logical access to grab relevant array elements
        train_samples_norm = train_samples_norm[train_logical_indices]
        test_samples_norm = test_samples_norm[test_logical_indices]
        train_labels = train_labels[train_logical_indices]
        test_labels = test_labels[test_logical_indices]
        num_total = (len(train_samples) + len(test_samples))
        num_dropout = (len(train_samples_norm) + len(test_samples_norm)) / num_total
        print(f'\na:{amount_t} d:{drop_t} :: {num_dropout} of {num_total}')

    # create onehot vectors out of labels
    train_labels = transform_labels(train_labels, cell=targets)
    test_labels = transform_labels(test_labels, cell=targets)
    train_labels = train_labels[:,targets].astype(int)
    test_labels = test_labels[:,targets].astype(int)
    train_labels_onehot = to_categorical(train_labels, num_classes)
    test_labels_onehot = to_categorical(test_labels, num_classes)

    epoch_steps = train_samples.shape[0] // batch_size

    if not return_tf_dataset:
        return train_samples_norm, test_samples_norm, train_labels, test_labels, epoch_steps, data_str
    else:
        # use Dataset as input pipeline
        train_data = tf.data.Dataset.from_tensor_slices((train_samples_norm, train_labels_onehot)).shuffle(10000).batch(batch_size, drop_remainder=True).repeat(-1)
        test_data  = tf.data.Dataset.from_tensor_slices((test_samples_norm, test_labels_onehot)).batch(batch_size)
        return train_data, test_data, train_labels, test_labels, epoch_steps, data_str

def prepare_dataset_mixture(targets, num_classes, mixture, batch_size=64):
    train_samples_hh, train_labels_hh, test_samples_hh, test_labels_hh = np.load('/home/ben/Desktop/ML/pretty_data/final.npy', allow_pickle=True)
    train_samples_syn, train_labels_syn, test_samples_syn, test_labels_syn = np.load('/home/ben/Desktop/ML/synthetic_data/final.npy', allow_pickle=True)

    hh_counts = dict(zip(*np.unique(train_labels_hh[:,targets].astype(int), return_counts=True)))
    syn_counts = dict(zip(*np.unique(train_labels_syn[:,targets].astype(int), return_counts=True)))

    for (hhl,hhc),(_,sync) in zip(hh_counts.items(), syn_counts.items()):
        n = int(hhc * mixture)
        if n > sync:
            print(f'Warning: Requested amount of data ({n}) for label {hhl} is larger than synthetic data for this label ({sync})')
        indices = (train_labels_syn[:,targets].astype(int) == hhl)
        train_samples_hh = np.vstack((train_samples_hh, train_samples_syn[indices][:n]))
        train_labels_hh = np.vstack((train_labels_hh, train_labels_syn[indices][:n]))

    # test new dataset consistency
    # hh_counts_post = dict(zip(*np.unique(train_labels_hh[:,targets].astype(int), return_counts=True)))
    # syn_counts_post = dict(zip(*np.unique(train_labels_syn[:,targets].astype(int), return_counts=True)))
    # for (hhl1,hhc1),(hhl2,hhc2) in zip(hh_counts_post.items(), hh_counts.items()):
    #     print(f'label: {hhl1}, before: {hhc2}, after: {hhc1}, diff: {hhc1-hhc2}')

    # normalise each sample with its own np.max
    train_samples_norm = normalise_minmax(train_samples_hh)
    test_samples_norm = normalise_minmax(test_samples_hh)
    # create onehot vectors out of labels
    train_labels = transform_labels(train_labels_hh, cell=targets)
    test_labels = transform_labels(test_labels_hh, cell=targets)
    train_labels = train_labels_hh[:,targets].astype(int)
    test_labels = test_labels_hh[:,targets].astype(int)
    train_labels_onehot = to_categorical(train_labels, num_classes)
    test_labels_onehot = to_categorical(test_labels, num_classes)
    # use Dataset as input pipeline
    train_data = tf.data.Dataset.from_tensor_slices((train_samples_norm, train_labels_onehot)).shuffle(10000).batch(batch_size, drop_remainder=True).repeat(-1)
    test_data  = tf.data.Dataset.from_tensor_slices((test_samples_norm, test_labels_onehot)).batch(batch_size)

    # class distribution
    class_counts = dict(zip(*np.unique(train_labels, return_counts=True)))
    print('current class distribution: ', class_counts)

    epoch_steps = train_samples_norm.shape[0] // batch_size
    data_str = f'handheld data augmented with {mixture*100}% synthetic data'

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
        net = Dense(64, activation='relu', kernel_regularizer=reg(reg_lambda))(inputs)
        net = Dropout(0.5)(net)
        net = Dense(256, activation='relu', kernel_regularizer=reg(reg_lambda))(net)
        net = Dropout(0.5)(net)
        net = Dense(256, activation='relu', kernel_regularizer=reg(reg_lambda))(net)
        # net = Dropout(0.5)(net)
        net = Dense(num_classes, activation='softmax')(net)

    model = Model(inputs=inputs, outputs=net, name=model_name)
    return model

def build_model_concat(id, num_classes, inputs=None, new_input=False, concat_model=None, reg=regularizers.l2, reg_lambda=0.0001):
    model_name = name if name else f'model_{id}'
    with tf.name_scope(model_name):
        if new_input:
            inputs = Input(shape=(7810,))
        if concat_model:
            inputs = Concatenate()([concat_model.inputs[0], concat_model.layers[-2].output])
        net = Dense(64, activation='relu', kernel_regularizer=reg(reg_lambda))(inputs)
        net = Dropout(0.5)(net)
        net = Dense(256, activation='relu', kernel_regularizer=reg(reg_lambda))(net)
        net = Dropout(0.5)(net)
        net = Dense(256, activation='relu', kernel_regularizer=reg(reg_lambda))(net)
        net = Dropout(0.5)(net)
        if not concat_model:
            net = Dense(7810, activation='relu', kernel_regularizer=reg(reg_lambda))(net)
            # net = Dropout(0.5)(net)
        net = Dense(num_classes, activation='softmax')(net)

    if not concat_model:
        model = Model(inputs=inputs, outputs=net, name=model_name)
    else:
        model = Model(inputs=concat_model.inputs[0], outputs=net, name=model_name)
    return model