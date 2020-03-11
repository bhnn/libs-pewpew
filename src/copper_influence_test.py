import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score)
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from utils import build_model, prepare_dataset, set_classification_targets


def classify(**args):
    overall_res = list()
    for amount_t in tqdm([.1, .2, .3, .4, .5, .6, .7, .8, .9]):
        res_per_amount = list()
        for drop_t in tqdm([.1, .2, .3, .4, .5, .6, .7, .8, .9], leave=False):
            batch_size = 64
            # determine classification targets and parameters to construct datasets properly
            num_classes, cls_target, cls_str = set_classification_targets(args['cls_choice'])
            train_data, test_data, train_labels, test_labels, epoch_steps, data_str = prepare_dataset(
                args['dataset_choice'], cls_target, num_classes, batch_size, amount_t=amount_t, drop_t=drop_t)

            if len(train_labels) == 0 or len(np.unique(train_labels)) != 12:
                res_per_amount.append([0.])
                continue

            # list of "class" names used for confusion matrices and validity testing. Not always classes, also subgroups or minerals
            class_names = [i for i in range(num_classes)]
            # class_weights = class_weight.compute_class_weight(
            #     'balanced', class_names, train_labels)

            # print(f'\n\tTask: Classify «{cls_str}» using «{data_str}»\n')

            model = build_model(
                0, num_classes, name='baseline_mlp', new_input=True)
            model.compile(optimizer=Adam(learning_rate=0.001, amsgrad=False),
                          loss='categorical_crossentropy', metrics=['accuracy'])

            # train and evaluate
            model.fit(train_data, steps_per_epoch=epoch_steps, epochs=5,
                      verbose=0, use_multiprocessing=True, ) #class_weight=class_weights)
            model.evaluate(test_data, verbose=0, use_multiprocessing=True)

            # predict on testset and calculate classification report and confusion matrix for diagnosis
            pred = model.predict(test_data, use_multiprocessing=True)
            pred = pred.argmax(axis=1)
            res = [balanced_accuracy_score(test_labels, pred)]#, precision_score(
                # test_labels, pred, average='samples'), recall_score(test_labels, pred, average='samples')]
            res_per_amount.append(res)
        overall_res.append(res_per_amount)

    l = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    for i in range(len(overall_res)):
        for j in range(len(overall_res[0])):
            print(f'{l[i]:3}\t{l[j]:3}\t{overall_res[i][j]}')

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
