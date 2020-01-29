import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, Average
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from utils import normalise_minmax, transform_labels

def build_model(id, inputs, num_classes, new_input=False):
    with tf.name_scope(f'model_{id}'):
        if new_input:
            inputs = Input(shape=(7810,))
        net = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(inputs)
        net = Dropout(0.5)(net)
        net = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(net)
        net = Dropout(0.5)(net)
        net = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(net)
        net = Dropout(0.5)(net)
        net = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(net)
        net = Dropout(0.5)(net)
        net = Dense(num_classes, activation='softmax')(net)

    model = keras.Model(inputs=inputs, outputs=net)
    return model

def classify(**args):
    # load different datasets
    if args['dataset_choice'] == 0:
        train_samples, train_labels, test_samples, test_labels = np.load('/home/ben/Desktop/ML/synthetic_data/final.npy', allow_pickle=True)
        data_str = 'synthetic data'
    elif args['dataset_choice'] == 1:
        train_samples, train_labels, test_samples, test_labels = np.load('/home/ben/Desktop/ML/pretty_data/final.npy', allow_pickle=True)
        data_str = 'handheld data'
    elif args['dataset_choice'] == 2:
        train_samples_1, train_labels_1, test_samples_1, test_labels_1 = np.load('/home/ben/Desktop/ML/synthetic_data/final.npy', allow_pickle=True)
        train_samples_2, train_labels_2, test_samples_2, test_labels_2 = np.load('/home/ben/Desktop/ML/pretty_data/final.npy', allow_pickle=True)
        train_samples = np.vstack((train_samples_1, train_samples_2))
        test_samples = np.vstack((test_samples_1, test_samples_2))
        train_labels = np.vstack((train_labels_1, train_labels_2))
        test_labels = np.vstack((test_labels_1, test_labels_2))
        data_str = 'both synthetic and handheld data'
    else:
        raise ValueError('Invalid dataset parameter')

    # determine classification targets and parameters to construct labels properly
    if args['cls_choice'] == 0:
        num_classes = 4
        cls_target= 0
        cls_str = 'mineral classes'
    elif args['cls_choice'] == 1:
        num_classes = 5
        cls_target = 1
        cls_str = 'mineral subgroups'
    elif args['cls_choice'] == 2:
        num_classes = 12
        cls_target = 2
        cls_str = 'minerals'
    else:
        raise ValueError('Invalid classification target parameter')
    
    print(f'\n\tTask: Classify «{cls_str}» using «{data_str}» with «multiple MLPs»\n')

    # list of "class" names used for confusion matrices and validity testing. Not always classes, also subgroups or minerals
    class_names = [i for i in range(num_classes)]
    batch_size = 64

    # normalise each sample with its own np.max
    train_samples_norm = normalise_minmax(train_samples)
    test_samples_norm = normalise_minmax(test_samples)
    # create onehot vectors out of labels
    train_labels = transform_labels(train_labels, cell=cls_target)
    test_labels = transform_labels(test_labels, cell=cls_target)
    train_labels = train_labels[:,cls_target].astype(int)
    test_labels = test_labels[:,cls_target].astype(int)
    train_labels_onehot = keras.utils.to_categorical(train_labels, num_classes)
    test_labels_onehot = keras.utils.to_categorical(test_labels, num_classes)
    # use Dataset as input pipeline
    train_data = tf.data.Dataset.from_tensor_slices((train_samples_norm, train_labels_onehot)).shuffle(10000).batch(batch_size, drop_remainder=True).repeat(-1)
    test_data  = tf.data.Dataset.from_tensor_slices((test_samples_norm, test_labels_onehot)).batch(batch_size)

    inputs = Input(shape=(7810,))
    models = [build_model(i, inputs, num_classes) for i in range(3)]

    multi_output = [m.outputs[0] for m in models]
    y = Average()(multi_output)
    model = keras.Model(inputs, outputs=y, name='ensemble')

    class_weights = class_weight.compute_class_weight('balanced', class_names, train_labels)
    epoch_steps = train_samples.shape[0] // batch_size

    tb_callback = keras.callbacks.TensorBoard(log_dir='./results', histogram_freq=0, write_graph=True, write_images=True)

    model.compile(optimizer=Adam(learning_rate=0.001, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, steps_per_epoch=epoch_steps, epochs=5, verbose=1, callbacks=[tb_callback], class_weight=class_weights, use_multiprocessing=True)
    model.evaluate(test_data, verbose=1, use_multiprocessing=True)

    plot_model(model, to_file='multiple_mlp.png')

    pred = model.predict(test_data, use_multiprocessing=True)
    print(classification_report(y_true=test_labels, y_pred=pred.argmax(axis=1), labels=class_names))
    matrix = confusion_matrix(y_true=test_labels, y_pred=pred.argmax(axis=1), labels=class_names)

    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize = (10,7))
    sn.heatmap(matrix, annot=True, fmt='.2f')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        type=int,
        default=0,
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

# class_names =           [0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
# actual_class_names =    [11, 19, 26, 28, 35, 41, 73, 80, 86, 88, 97, 98]
# carbonates    - 11, 73, 88
# oxides        - 41, 28, 97
# phosphates    - 80, 86, 35
# sulfides      - 26, 98, 19

# todo - mit Callback Early Stopping so lang laufen lassen, bis Val-Acc nach 10 Epochen nicht mehr sinkt