import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Input, Maximum
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
from utils import build_model, prepare_dataset, set_classification_targets


def classify(**args):
    batch_size = 64
    # determine classification targets and parameters to construct datasets properly
    num_classes, cls_target, cls_str = set_classification_targets(args['cls_choice'])
    train_data, test_data, train_labels, test_labels, epoch_steps, data_str = prepare_dataset(0, cls_target, num_classes, batch_size)

    # list of "class" names used for confusion matrices and validity testing. Not always classes, also subgroups or minerals
    class_names = [i for i in range(num_classes)]
    # class_weights = class_weight.compute_class_weight('balanced', class_names, train_labels)
    
    print(f'\n\tTask: Classify «{cls_str}» using «{data_str}»\n')

    # class weights: [0.503789 1.47479113 0.5920657 1.98873349 1.98873349 0.62365984 0.53289611 1.20418725 1.5526236 2.157185 1.86280932 1.45467462]
    class_weights = [
        [10, 0.01, 0.01, 0.01],
        [0.01, 10, 0.01, 0.01],
        [0.01, 0.01, 10, 0.01],
        [0.01, 0.01, 0.01, 10],
]

    models = list()
    inputs = Input(shape=(7810,))

    for i in range(4):
        model = build_model(i, num_classes, inputs=inputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, steps_per_epoch=epoch_steps, epochs=5, verbose=1, class_weight=class_weights[i], use_multiprocessing=True)
        models.append(model)

    multi_output = [m.outputs[0] for m in models]
    y = Maximum()(multi_output)
    model = Model(inputs, outputs=y, name='ensemble')
    plot_model(model, to_file='img/ensemble_mlp.png')

    tb_callback = TensorBoard(log_dir='./results', histogram_freq=0, write_graph=True, write_images=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.evaluate(test_data, verbose=1, use_multiprocessing=True)

    # predict on testset and calculate classification report and confusion matrix for diagnosis
    pred = model.predict(test_data, use_multiprocessing=True)
    print(classification_report(y_true=test_labels, y_pred=pred.argmax(axis=1), labels=class_names))

    # normalised confusion matrix
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
