import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from utils import prepare_dataset, set_classification_targets


def build_model(id, num_classes, inputs=None, new_input=False, concat_model=None, reg=regularizers.l2, reg_lambda=0.0001):
    with tf.name_scope(f'model_{id}'):
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

def classify(**args):
    # load different datasets
    batch_size = 64
    # determine classification targets and parameters to construct datasets properly
    num_classes, cls_target, cls_str = set_classification_targets(args['cls_choice'])
    train_data, test_data, train_labels, test_labels, epoch_steps, data_str = prepare_dataset(0, cls_target, num_classes, batch_size)

    print(f'\n\tTask: Classify «{cls_str}» using «{data_str}»\n')

    # list of "class" names used for confusion matrices and validity testing. Not always classes, also subgroups or minerals
    class_names = [i for i in range(num_classes)]
    class_weights = class_weight.compute_class_weight('balanced', class_names, train_labels)

    model = build_model(0, num_classes, new_input=True)

    model.summary()

    # tb_callback = keras.callbacks.TensorBoard(log_dir='./results', histogram_freq=0, write_graph=True, write_images=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, steps_per_epoch=epoch_steps, epochs=5, verbose=1, class_weight=class_weights, use_multiprocessing=True)
    model.evaluate(test_data, verbose=1, use_multiprocessing=True)

    num_classes, cls_target, cls_str = set_classification_targets(args['cls_choice'])
    train_data_hh, test_data_hh, train_labels_hh, test_labels_hh, epoch_steps_hh, data_str = prepare_dataset(1, cls_target, num_classes, batch_size)
    class_weights_hh = class_weight.compute_class_weight('balanced', class_names, train_labels)

    concat_model = build_model(1, num_classes, concat_model=model)
    concat_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    concat_model.fit(train_data_hh, steps_per_epoch=epoch_steps_hh, epochs=5, verbose=1, class_weight=class_weights_hh, use_multiprocessing=True)

    concat_model.summary()

    plot_model(concat_model, to_file='img/concat_mlp.png')

    pred = concat_model.predict(test_data_hh, use_multiprocessing=True)
    print(classification_report(y_true=test_labels_hh, y_pred=pred.argmax(axis=1), labels=class_names))
    matrix = confusion_matrix(y_true=test_labels_hh, y_pred=pred.argmax(axis=1), labels=class_names)

    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize = (10,7))
    sn.heatmap(matrix, annot=True, fmt='.2f')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
