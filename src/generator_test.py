import argparse
import sys
from glob import glob
from json import dumps
from ntpath import basename
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight, shuffle
from tensorflow.keras.optimizers import Adam

from utils import build_model


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
    print(classification_report(y_true, y_pred.argmax(axis=1), labels=class_ids))

    # normalised confusion matrix
    matrix = confusion_matrix(y_true, y_pred.argmax(axis=1), labels=class_ids)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize = (10,7))
    sn.heatmap(matrix, annot=True, fmt='.2f')
    plt.gca().tick_params(axis='y', rotation=45)
    plt.show()

def __data_generator(files, targets, num_classes, batch_size, trans_dict, shuffle_and_repeat):
    """
    Internal generator function to yield processed batches of data.
    
    :param files:           list of files to work with
    :param targets:         classification targets chosen by user
    :param num_classes:     number of unique classes in dataset
    :param batch_size:      batch size of yielded batches
    :param trans_dict:      transformation info to process labels
    :shuffle_and_repeat:    True to shuffle and repeat dataset infinitely, False otherwise
    :yields: (sample, label) tuples
    """
    i = 0
    run = True
    files = shuffle(files) if shuffle_and_repeat else files
    while run:
        # samples: 7810 data points per file, labels: num_classes size of categorical labels
        samples = np.zeros((batch_size, 7810), dtype='float64')
        labels = np.zeros((batch_size, num_classes), dtype=int)
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
                labels[j]  = to_categorical(transform_labels(npz_file['labels'][targets], trans_dict), num_classes)
                samples[j] = normalise_minmax(npz_file['data'])
            i += 1
        yield samples, labels

def prepare_dataset_test(dataset_choice, target, batch_size):
    """
    Provides data generators, labels and other information for selected dataset. 
    
    :param dataset_choice:  which dataset to prepare
    :param targets:         classification target
    :param batch_size:      batch size
    :returns:   dict containing train/eval/test generators, train/test labels, number of unique classes, 
                original and transformed class ids, train/test steps, balanced class weights and data description
    """
    if dataset_choice == 0:
        data_path = r"/media/ben/Volume/ml_data/synthetic"
        data_str = 'synthetic data'
    elif dataset_choice == 1:
        data_path = r'/home/ben/Desktop/ML/hh_6'
        data_str = 'handheld data (6 classes)'
    elif dataset_choice == 2:
        data_path = r'/media/ben/Volume/ml_data/hh_raw/hh_12'
        data_str = 'handheld data (12 classes)'
    elif dataset_choice == 3:
        data_path = r'/media/ben/Volume/ml_data/hh_raw/hh_all'
        data_str = 'handheld data (100 classes)'
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
        'train_data'   : __data_generator(train_data, target, num_classes, batch_size, trans_dict, True),
        'eval_data'    : __data_generator(test_data, target, num_classes, batch_size, trans_dict, False),
        'test_data'    : __data_generator(test_data, target, num_classes, batch_size, trans_dict, False),
        'train_labels' : transform_labels(train_labels, trans_dict),
        'test_labels'  : transform_labels(test_labels, trans_dict),
        'num_classes'  : num_classes,
        'classes_orig' : sorted(trans_dict.keys()),
        'classes_trans': sorted(trans_dict.values()),
        'train_steps'  : len(train_labels) // batch_size,
        'test_steps'   : (len(test_labels) // batch_size)+1,
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
        print('\t\t{:<13} : {},{}'.format(k, v, f' len({len(v)}),' if hasattr(v, '__len__') else ''))
    print('\t}\n')

def classify(**args):
    """
    Main method that prepares dataset, builds model, executes training and displays results.
    
    :param args: keyword arguments passed from cli parser
    """
    batch_size = 64
    # determine classification targets and parameters to construct datasets properly
    cls_target, cls_str = set_classification_targets(args['cls_choice'])
    d = prepare_dataset_test(args['dataset_choice'], cls_target, batch_size)

    print('\n\tTask: Classify «{}» using «{}»\n'.format(cls_str, d['data_str']))
    print_dataset_info(d)

    model = build_model(0, d['num_classes'], name='baseline_mlp', new_input=True)
    model.compile(optimizer=Adam(learning_rate=0.001, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])

    # train and evaluate
    model.fit(d['train_data'], steps_per_epoch=d['train_steps'], epochs=args['epochs'], verbose=1, class_weight=d['class_weights'])
    model.evaluate(d['eval_data'], steps=d['test_steps'], verbose=1)

    # predict on testset and calculate classification report and confusion matrix for diagnosis
    pred = model.predict(d['test_data'], steps=d['test_steps'])

    diagnose_output(d['test_labels'], pred, d['classes_trans'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        type=int,
        default=1,
        help='Which dataset(s) to use. 0=synthetic, 1=handheld, 2=both',
        dest='dataset_choice'
    )
    parser.add_argument(
        '-c', '--classification',
        type=int,
        default=2,
        help='Which classification target to pursue. 0=classes, 1=subgroups, 2=minerals',
        dest='cls_choice'
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=5,
        help='How many epochs to train for',
        dest='epochs'
    )
    args = parser.parse_args()

    classify(**vars(args))
