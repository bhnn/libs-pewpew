import argparse

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from utils import (build_model, diagnose_output, prepare_mixture_dataset,
                   print_dataset_info, repeat_and_collate,
                   set_classification_targets)
import os
from os.path import join

path = r'/Users/jh/github'


def classify(**args):
    batch_size = 64
    repetitions = args['repetitions']
    # determine classification targets and parameters to construct datasets properly
    cls_target, cls_str = set_classification_targets(args['cls_choice'])

    # list list of 5% increments ranging from 0% to 100%
    mixture_range = np.arange(0, 1.01, .05)
    results = np.zeros((len(mixture_range), repetitions))

    for i,cut in enumerate(mixture_range):
        print(f'cut: {cut}')
        d = prepare_mixture_dataset(
        cls_target,
        args['batch_size'],
        mixture_pct=cut,
        args['norm_choice'])

        for j in range(repetitions):
            model = build_model(0, d['num_classes'], name='baseline_mlp', new_input=True)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # train and evaluate
            model.fit(
                d['train_data'],
                steps_per_epoch=d['train_steps'],
                epochs=args['epochs'],
                verbose=0,
                class_weight=d['class_weights'])
            # evaluate returns (final loss, final acc), thus the [1]
            results[i,j] = model.evaluate(d['test_data'](), steps=d['test_steps'], verbose=1)[1]
    print(results)
    np.save(os.path.join(path, 'libs-pewpew/data/synthetic_influence_target_{cls_target}', results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--repetitions',
        type=int,
        default=1,
        help='Number of times to repeat experiment',
        dest='repetitions'
    )
    parser.add_argument(
        '-b', '--batchsize',
        type=int,
        default=64,
        help='Target batch size of dataset preprocessing',
        dest='batch_size'
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
    parser.add_argument(
        '-n', '--normalisation',
        type=int,
        default=2,
        help='Which normalisation to use. 0=None, 1=snv, 2=minmax',
        dest='norm_choice'
    )
    args = parser.parse_args()

    classify(**vars(args))
