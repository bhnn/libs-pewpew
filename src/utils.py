from glob import glob
from math import ceil
from ntpath import basename
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight, shuffle
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input


def has_sufficient_copper(spectrum, amount_t=0.5, drop_t=0.3):
    """
    Checks individual spectras for their copper content. If more than *amount_t*% of copper lines are below the *drop_t*
    percentile, then the spectrum more likely captures matrix rock.
    
    :param spectrum:    ndarray containing the spectrum
    :param amount_t:    threshold (float %) of copper lines expected to have high intensity
    :param drop_t:      threshold (float %) signifying "low" copper intensity
    :returns: True if the spectrum contains more than the specified amount of copper, False otherwise
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

def set_classification_targets(cls_choice):
    """
    Validate classification choice by user and provide written detail about what the goal of the classification effort is.
    
    :param cls_choice: Classification choice provided by user (via cli parameter)
    :returns: tuple of (classification choice, classification string)
    :raise ValueError: cls_choice is out of defined range
    """
    if cls_choice == 0:
        cls_str = 'mineral classes'
    elif cls_choice == 1:
        cls_str = 'mineral subgroups'
    elif cls_choice == 2:
        cls_str = 'minerals'
    else:
        raise ValueError('Invalid classification target parameter')
    return cls_choice, cls_str

def __get_labels(files, targets):
    """
    Extracts data labels from filenames of passed list of filepaths. Faster than loading all files into memory. Only
    provides labels for current classification target.
    
    :param files:       list of file paths
    :param targets:     classification targets
    :returns:           ndarray(int) of labels 
    :raises error type: raises description
    """
    # translates targets into position of information in filename
    target_indices = [1, 2, 0]
    return np.array([int(basename(f).split('_')[target_indices[targets]]) for f in files], dtype=int)

def get_transformation_dict(labels, dict_or_np='dict'):
    """
    Creates gapless list of labels (e.g. 0, 1, 2, 3, 4) from "sparse" data labels (e.g. 11, 28, 35, 73, 98).
    
    :param labels:      list of labels to transform
    :param dict_or_np:  {'dict', 'np', 'numpy'} whether returned transformation matrix should be a dict or ndarray
    :returns:           label transformation info, as dict or ndarray
    """
    trans_dict = {unique_label : i for i, unique_label in enumerate(np.unique(labels))}
    if dict_or_np == 'np' or dict_or_np == 'numpy':
        trans_dict = np.array([trans_dict[i] if i in trans_dict.keys() else 0 for i in range(np.max(np.unique(labels))+1)])
    return trans_dict

def transform_labels(label_data, trans_dict):
    """
    Transforms labels according to information from trans_dict.
    
    :param label_data:  labels to transform
    :param trans_dict:  transformation info, as dict or ndarray
    :returns:           list of transformed labels
    """
    if isinstance(label_data, (list, np.ndarray)):
        return np.array([trans_dict[l] for l in label_data])
    else:
        return trans_dict[label_data]

def normalise_minmax(sample):
    """
    Normalises single sample according to minmax. Also strips wavelength information from sample.
    Adapted from Federico Malerba.
    
    :param sample:  sample to process
    :returns:       normalised sample
    """
    if np.max(sample[:,1]) > 0:
        # only work with 2nd column of information (intensity), discard wavelength
        sample = sample[:,1] / np.max(sample[:,1])
    else:
        sample = np.zeros(sample.shape[0])
    return sample

def diagnose_output(y_true, y_pred, class_ids):
    """
    Calculates sklearn.metrics.classification_report and confusion matrix for provided data and visualises them.
    
    :param y_true:      true label information
    :param y_pred:      predicted values
    :param class_ids:   transformed ids of classes for plotting
    """
    print(classification_report(y_true, y_pred, labels=class_ids))

    # normalised confusion matrix
    matrix = confusion_matrix(y_true, y_pred, labels=class_ids)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize = (10,7))
    sn.heatmap(matrix, annot=True, fmt='.2f') # xticklabels=target_names, yticklabels=target_names)
    plt.gca().tick_params(axis='y', rotation=45)
    plt.title('Confusion matrix (normalised)')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

def __data_generator(files, targets, num_classes, batch_size, trans_dict, shuffle_and_repeat, categorical=True):
    """
    Internal generator function to yield processed batches of data.
    
    :param files:               list of files to work with
    :param targets:             classification targets chosen by user
    :param num_classes:         number of unique classes in dataset
    :param batch_size:          batch size of yielded batches
    :param trans_dict:          transformation info to process labels
    :param shuffle_and_repeat:  True to shuffle and repeat dataset infinitely, False otherwise
    :param categorical:         True to provide categorical labels, False otherwise
    :yields:                    (sample, label) tuples
    """
    i = 0
    run = True
    files = shuffle(files) if shuffle_and_repeat else files
    while run:
        # samples: 7810 data points per file, labels: num_classes size of categorical labels
        samples = np.zeros((batch_size, 7810), dtype='float64')
        labels = np.zeros((batch_size, num_classes), dtype=int) if categorical else np.zeros((batch_size,), dtype=int)
        # read in batch_size many np files
        for j in range(batch_size):
            # after all files have been consumed
            if i >= len(files):
                if shuffle_and_repeat:
                    i = 0
                    files = shuffle(files)
                else:
                    run = False
                    samples, labels = samples[:j], labels[:j]
                    break

            # load data, all:(data:(7810, 2), labels:(3,))
            with np.load(files[i]) as npz_file:
                label = transform_labels(npz_file['labels'][targets], trans_dict)
                labels[j]  = to_categorical(label, num_classes) if categorical else label
                samples[j] = normalise_minmax(npz_file['data'])
            i += 1
        yield samples, labels

def prepare_dataset(dataset_choice, target, batch_size, train_shuffle_repeat=True, categorical_labels=True):
    """
    Provides data generators, labels and other information for selected dataset. 
    
    :param dataset_choice:          which dataset to prepare
    :param targets:                 classification target
    :param batch_size:              batch size
    :param train_shuffle_repeat:    whether to shuffle and repeat the train generator
    :param categorical_labels:      whether to transform labels to categorical
    :returns:                       dict containing train/eval/test generators, train/test labels, number of unique 
                                    classes, original and transformed class ids, train/test steps, balanced class 
                                    weights and data description
    :raises ValueError:             if dataset_choice is invalid
    """
    if dataset_choice == 0:
        data_path = r"/media/ben/Volume/ml_data/synthetic"
        data_str = 'synthetic data'
        data_name = 'synthetic'
    elif dataset_choice == 1:
        data_path = r'/home/ben/Desktop/ML/hh_6'
        data_str = 'handheld data (6 classes)'
        data_name = 'hh_6'
    elif dataset_choice == 2:
        data_path = r'/media/ben/Volume/ml_data/hh_raw/hh_12'
        data_str = 'handheld data (12 classes)'
        data_name = 'hh_12'
    elif dataset_choice == 3:
        data_path = r'/media/ben/Volume/ml_data/hh_raw/hh_all'
        data_str = 'handheld data (100 classes)'
        data_name = 'hh_all'
    else:
        raise ValueError('Invalid dataset parameter')
    
    train_data = sorted(glob(join(data_path, 'train', '*.npz')))
    test_data = sorted(glob(join(data_path, 'test', '*.npz')))

    train_labels = __get_labels(train_data, target)
    test_labels = __get_labels(test_data, target)

    num_classes = len(np.unique(train_labels))
    trans_dict = get_transformation_dict(train_labels)
    class_weights = class_weight.compute_class_weight('balanced', sorted(trans_dict.keys()), train_labels)

    # list of "class" names used for confusion matrices and validity testing. Not always classes, also subgroups or
    # minerals, depending on classification targets
    return {
        'dataset_name' : data_name,
        'train_data'   : __data_generator(train_data, target, num_classes, batch_size, trans_dict, train_shuffle_repeat, categorical_labels),
        'eval_data'    : __data_generator(test_data, target, num_classes, batch_size, trans_dict, False, categorical_labels),
        'test_data'    : __data_generator(test_data, target, num_classes, batch_size, trans_dict, False, categorical_labels),
        'train_labels' : transform_labels(train_labels, trans_dict),
        'test_labels'  : transform_labels(test_labels, trans_dict),
        'num_classes'  : num_classes,
        'classes_orig' : sorted(trans_dict.keys()),
        'classes_trans': sorted(trans_dict.values()),
        'train_steps'  : ceil(len(train_labels) / batch_size),
        'test_steps'   : ceil(len(test_labels) / batch_size),
        'class_weights': {i:weight for i, weight in enumerate(class_weights)},
        'data_str'     : data_str,
    }

def print_dataset_info(dataset):
    """
    Prints formatted dataset information for the user.
    
    :param dataset: dict containing dataset information
    """
    print('\tData set information:\n\t{')
    for k,v in dataset.items():
        print('\t\t{:<13} : {},{}'.format(k, v, f' len({len(v)}),' if hasattr(v, '__len__') and not isinstance(v, str) else ''))
    print('\t}\n')

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
    model_name = f'model_{id}'
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
