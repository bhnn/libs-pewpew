import argparse
import sys
from glob import glob
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
    # translates targets into position of information in filename
    target_indices = [1, 2, 0]
    return np.array([int(basename(f).split('_')[target_indices[targets]]) for f in files], dtype=int)

def get_transformation_dict(labels, dict_or_np='dict'):
    trans_dict = {unique_label : i for i, unique_label in enumerate(np.unique(labels))}
    if dict_or_np == 'np' or dict_or_np == 'numpy':
        trans_dict = np.array([trans_dict[i] if i in trans_dict.keys() else 0 for i in range(np.max(np.unique(labels))+1)])
    return trans_dict

def transform_labels(label_data, trans_dict):
    if isinstance(label_data, (list, np.ndarray)):
        return np.array([trans_dict[l] for l in label_data])
    else:
        return trans_dict[label_data]

def normalise_minmax(sample):
    if np.max(sample[:,1]) > 0:
        sample = sample[:,1] / np.max(sample[:,1])
    else:
        sample = np.zeros(sample.shape[0])
    return sample

def __data_generator(files, targets, num_classes, batch_size, trans_dict, shuffle_and_repeat):
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
                    samples, labels = samples[:j-1], labels[:j-1]
                    break

            # load data, all:(data:(7810, 2), labels:(3,))
            with np.load(files[i]) as npz_file:
                labels[j]  = to_categorical(transform_labels(npz_file['labels'][targets], trans_dict), num_classes)
                samples[j] = normalise_minmax(npz_file['data'])
            i += 1
        yield samples, labels

def prepare_dataset_test(dataset_choice, targets, batch_size):
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
    
    train_data   = sorted(glob(join(data_path, 'train', '*.npz')))
    train_labels = __get_labels(train_data, targets)

    test_data    = sorted(glob(join(data_path, 'test', '*.npz')))
    test_labels  = __get_labels(test_data, targets)

    num_classes = len(np.unique(train_labels))
    trans_dict  = get_transformation_dict(train_labels)
    train_labels = transform_labels(train_labels, trans_dict)
    test_labels  = transform_labels(test_labels, trans_dict)

    train_gen = __data_generator(train_data, targets, num_classes, batch_size, trans_dict, True)
    test_gen  = __data_generator(test_data, targets, num_classes, batch_size, trans_dict, False)

    # for d0, d1 in train_gen:
    #     print(d1)
    #     with np.load(path) as npz:
    #         t0 = npz['data']
    #         t1 = npz['labels']
    #     assert d0 == t0, 'sample wrong'
    #     assert d1 == t1, 'label wrong'

    # t0 = np.array([normalise_minmax(npz['data']) for npz in [np.load(f) for f in test_data[:64]]])
    # t1 = np.array([to_categorical(transform_labels(npz['labels'], trans_dict), num_classes) for npz in [np.load(f) for f in test_data[:64]]])
    # for d0,d1 in test_gen:
    #     print(np.count_nonzero(d0[0]))
    #     print(np.count_nonzero(t0[0]))
    #     break

    epoch_steps = (len(train_labels) // batch_size, len(test_labels) // batch_size)
    return train_gen, test_gen, train_labels, test_labels, epoch_steps, num_classes, data_str

def classify(**args):
    batch_size = 64
    # determine classification targets and parameters to construct datasets properly
    cls_target, cls_str = set_classification_targets(args['cls_choice'])
    train_data, test_data, train_labels, test_labels, epoch_steps, num_classes, data_str = prepare_dataset_test(args['dataset_choice'], cls_target, batch_size)

    # list of "class" names used for confusion matrices and validity testing. Not always classes, also subgroups or minerals
    class_names = [i for i in range(num_classes)]
    class_weights = class_weight.compute_class_weight('balanced', class_names, train_labels)
    weight_dict = {i:weight for i,weight in enumerate(class_weights)}
    print('class weights:', weight_dict)
    
    print(f'\n\tTask: Classify «{cls_str}» using «{data_str}»\n')

    model = build_model(0, num_classes, name='baseline_mlp', new_input=True)
    model.compile(optimizer=Adam(learning_rate=0.001, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])

    # train and evaluate
    model.fit(train_data, steps_per_epoch=epoch_steps[0], epochs=5, verbose=1, class_weight=weight_dict, use_multiprocessing=False)
    model.evaluate(test_data, steps=epoch_steps[1]+1, verbose=1, use_multiprocessing=False)

    sys.exit(1)

    # predict on testset and calculate classification report and confusion matrix for diagnosis
    pred = model.predict(test_data, use_multiprocessing=False)
    print(pred.shape)
    print(classification_report(y_true=test_labels, y_pred=pred.argmax(axis=1), labels=class_names))

    # normalised confusion matrix
    matrix = confusion_matrix(y_true=test_labels, y_pred=pred.argmax(axis=1), labels=class_names)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    print(matrix.shape)

    plt.figure(figsize = (10,7))
    sn.heatmap(matrix, annot=True, fmt='.2f')
    plt.show()

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
        dest='cls_choice')
    args = parser.parse_args()
    
    classify(**vars(args))
