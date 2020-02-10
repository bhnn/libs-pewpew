import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from utils import build_model, prepare_dataset, prepare_dataset_mixture, set_classification_targets


def classify(**args):
    batch_size = 64
    repetitions = 4
    # determine classification targets and parameters to construct datasets properly
    num_classes, cls_target, cls_str = set_classification_targets(args['cls_choice'])

    # list of "class" names used for confusion matrices and validity testing. Not always classes, also subgroups or minerals
    class_names = [i for i in range(num_classes)]
    
    mixture_range = np.arange(0, 1.01, .05)
    results = np.zeros((len(mixture_range), repetitions))

    for i,cut in enumerate(mixture_range):
        print(f'cut: {cut}')
        train_data, test_data, train_labels, test_labels, epoch_steps, _ = prepare_dataset_with_influence(cls_target, num_classes, cut, batch_size=batch_size)
        class_weights = class_weight.compute_class_weight('balanced', class_names, train_labels)

        for j in range(repetitions):
            model = build_model(0, num_classes, name='baseline_mlp', new_input=True)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

            # train and evaluate
            model.fit(train_data, steps_per_epoch=epoch_steps, epochs=10, verbose=0, class_weight=class_weights, use_multiprocessing=True)
            # evaluate returns (final loss, final acc), thus the [1]
            results[i,j] = model.evaluate(test_data, verbose=1, use_multiprocessing=True)[1]
            print(f'j: {j}')
    print(results)
    np.save(f'results/synthetic_influence_target_{cls_target}', results)

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
