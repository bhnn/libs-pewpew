import argparse
import sys
from glob import glob
from ntpath import basename
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight, shuffle

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
    return trans_dict[label_data]

def normalise_minmax(sample):
    if np.max(sample[:,1]) > 0:
        sample = sample[:,1] / np.max(sample[:,1])
    else:
        sample = np.zeros(sample.shape[0])
    return sample

def __load_npz(item, targets, num_classes, trans_dict):
    with np.load(item.numpy()) as npz:
        sample = normalise_minmax(npz['data'])
        label  = to_categorical(transform_labels(npz['labels'][targets], trans_dict), num_classes, dtype=np.int)
    return sample, label

def __data_generator(files, targets, num_classes, batch_size, trans_dict, shuffle_and_repeat, shuffle_buffer=3000):

    def __map_fn(elem):
        tensor = tuple(tf.py_function(__load_npz, [elem, targets, num_classes, trans_dict], [tf.float64, tf.int32]))
        # workaround: explicit shapes because true tensor values are only assigned by map function and dataset will not recognise
        tensor[0].set_shape(7810,)
        tensor[1].set_shape(num_classes)
        return tensor

    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(__map_fn)
    if shuffle_and_repeat:
        return dataset.shuffle(shuffle_buffer).batch(batch_size, drop_remainder=True).repeat(-1)#.as_numpy_iterator()
    else:
        return dataset.shuffle(shuffle_buffer).batch(batch_size, drop_remainder=True)#.as_numpy_iterator()

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
    trans_dict  = get_transformation_dict(train_labels, 'numpy')
    train_labels = transform_labels(train_labels, trans_dict)
    test_labels  = transform_labels(test_labels, trans_dict)

    train_gen = __data_generator(train_data, targets, num_classes, batch_size, trans_dict, True)
    test_gen  = __data_generator(test_data, targets, num_classes, batch_size, trans_dict, False)

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
    print('class weights:', class_weights, weight_dict)
    
    print(f'\n\tTask: Classify «{cls_str}» using «{data_str}»\n')

    model = build_model(0, num_classes, name='baseline_mlp', new_input=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train and evaluate
    model.fit(train_data, steps_per_epoch=epoch_steps[0], epochs=5, verbose=1, class_weight=weight_dict, use_multiprocessing=True)
    model.evaluate(test_data, steps=epoch_steps[1], verbose=1, use_multiprocessing=True)

    sys.exit(1)

    # predict on test set and calculate classification report and confusion matrix for diagnosis
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
        help='Which dataset(s) to use. 0=synthetic, 1=hh_6, 2=hh_12, 3=hh_all',
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
    # for i in range(1500):
    #     dim0 = np.round(np.arange(180, 961, 0.1), 1)
    #     dim1 = np.array([np.random.random() for _ in dim0])
    #     sample = np.stack((dim0, dim1), axis=1)
    #     label = np.array([np.random.randint(6) for _ in range(3)])
    #     np.savez_compressed(i, data=sample, labels=label)

    # print(sample.shape, label.shape)
