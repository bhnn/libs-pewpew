import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from utils import build_model, prepare_dataset, set_classification_targets


def classify(**args):
    batch_size = 64
    # determine classification targets and parameters to construct datasets properly
    print(args)
    num_classes, cls_target, cls_str = set_classification_targets(args['cls_choice'])
    train_data, test_data, train_labels, test_labels, epoch_steps, data_str = prepare_dataset(args['dataset_choice'], cls_target, num_classes, batch_size)

    # list of "class" names used for confusion matrices and validity testing. Not always classes, also subgroups or minerals
    class_names = [i for i in range(num_classes)]
    class_weights = class_weight.compute_class_weight('balanced', class_names, train_labels)
    print('class weights:', class_weights)
    
    print(f'\n\tTask: Classify «{cls_str}» using «{data_str}»\n')

    model = build_model(0, num_classes, name='baseline_mlp', new_input=True)
    model.compile(optimizer=Adam(learning_rate=0.001, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print('')
    plot_model(model, to_file='img/baseline_mlp.png')

    # callback to log data for TensorBoard
    tb_callback = TensorBoard(log_dir='./results', histogram_freq=0, write_graph=True, write_images=True)

    # train and evaluate
    model.fit(train_data, steps_per_epoch=epoch_steps, epochs=5, verbose=1, callbacks=[tb_callback], class_weight=class_weights, use_multiprocessing=True)
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
