import argparse
import sys
from glob import glob
from ntpath import basename
from os.path import join
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight, shuffle
from tensorflow.keras.optimizers import Adam

from utils import build_model, set_classification_targets


def transform_labels(label_data, transformation_dict=None):
    if not transformation_dict:
        transformation_dict = {unique_label : i for i, unique_label in enumerate(np.unique(label_data))}
    label_data = [transformation_dict[l] for l in label_data]
    return label_data, transformation_dict

def __data_generator(files, targets, num_classes, batch_size, transform_dict, shuffle_and_repeat):
    i = 0
    run = True
    files = shuffle(files) if shuffle_and_repeat else files
    while run:
        samples = np.zeros((batch_size, 7810), dtype='float64')
        labels = np.zeros((batch_size, 3), dtype=int)
        # read in batch_size many np files
        t0 = time()
        for j in range(batch_size):
            # after all files have been consumed
            if i >= len(files):
                if shuffle_and_repeat:
                    i = 0
                    files = shuffle(files)
                else:
                    run = False
                    break

            # load data, all:(data:(7810, 2), labels:(3,))
            t1 = time()
            sample, label = np.load(files[i], allow_pickle=True)
            t1_e = time()-t1
            if t1_e > 1:
                print('load', round(t1_e, 3), files[i])
            labels[j] = label
            # normalise data
            if np.max(sample[:,1]) > 0:
                samples[j] = sample[:,1] / np.max(sample[:,1])
            else:
                samples[j] = np.zeros(sample.shape[0])
            i += 1

        print('batch:', round(time()-t0, 3), '\n')
        # transform labels, strip away all but current targets, then convert to categorical
        labels, _ = transform_labels(labels[:, targets], transform_dict)
        labels = to_categorical(labels, num_classes)

        yield np.array(samples), np.array(labels)

def __get_labels(files, targets):
    # translates targets into position of information in filename
    target_indices = [1, 2, 0]
    labels = [int(basename(f).split('_')[target_indices[targets]]) for f in files]
    return transform_labels(labels)

def prepare_dataset_test(dataset_choice, targets, num_classes, batch_size=64):
    if dataset_choice == 0:
        data_path = r"/media/ben/Volume/ml_data/synthetic"
        data_str = 'synthetic data'
    elif dataset_choice == 1:
        data_path = r'/media/ben/Volume/ml_data/hh_12'
        data_str = 'handheld data'
    else:
        raise ValueError('Invalid dataset parameter')
    
    train_data = sorted(glob(join(data_path, 'train', '*.npy')))
    train_labels, transform_dict = __get_labels(train_data, targets)
    train_gen = __data_generator(train_data, targets, num_classes, batch_size, transform_dict, True)

    test_data = sorted(glob(join(data_path, 'test', '*.npy')))
    test_labels = __get_labels(test_data, targets)[0]
    test_gen = __data_generator(test_data, targets, num_classes, batch_size, transform_dict, False)

    epoch_steps = len(train_data) // batch_size
    return train_gen, test_gen, train_labels, test_labels, epoch_steps, data_str

def classify(**args):
    batch_size = 64
    # determine classification targets and parameters to construct datasets properly
    num_classes, cls_target, cls_str = set_classification_targets(args['cls_choice'])
    train_data, test_data, train_labels, test_labels, epoch_steps, data_str = prepare_dataset_test(args['dataset_choice'], cls_target, num_classes, batch_size)

    # list of "class" names used for confusion matrices and validity testing. Not always classes, also subgroups or minerals
    class_names = [i for i in range(num_classes)]
    class_weights = class_weight.compute_class_weight('balanced', class_names, train_labels)
    print('class weights:', class_weights)
    
    print(f'\n\tTask: Classify «{cls_str}» using «{data_str}»\n')

    model = build_model(0, num_classes, name='baseline_mlp', new_input=True)
    model.compile(optimizer=Adam(learning_rate=0.001, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])

    for d0, d1 in train_data:
        print(d0.shape, d1.shape)

    import sys
    sys.exit(1)

    # train and evaluate
    model.fit(train_data, steps_per_epoch=epoch_steps, epochs=5, verbose=1, class_weight=class_weights, use_multiprocessing=False)
    model.evaluate(test_data, verbose=1, use_multiprocessing=True)

    # predict on testset and calculate classification report and confusion matrix for diagnosis
    pred = model.predict(test_data, use_multiprocessing=True)
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
