import argparse
from os.path import join

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from utils import (build_model, diagnose_output, prepare_mixture_dataset,
                   print_dataset_info, repeat_and_collate,
                   set_classification_targets)

with open('config/datasets.yaml') as cnf:
    dataset_configs = yaml.safe_load(cnf)
    try:
        repo_path = dataset_configs['repo_path']
    except KeyError as e:
        print(f'Missing dataset config key: {e}')
        sys.exit(1)

def classify(**args):
    """
    Main method that prepares dataset, builds model, executes training and displays results.

    :param args: keyword arguments passed from cli parser
    """
    batch_size = 64
    repetitions = args['repetitions']
    # determine classification targets and parameters to construct datasets properly
    cls_target, cls_str = set_classification_targets(args['cls_choice'])

    # list of 5% increments ranging from 0% to 100%
    mixture_range = np.arange(0, 1.01, .05)
    results = np.zeros((len(mixture_range), repetitions))

    for i,cut in enumerate(mixture_range):
        print(f'cut: {cut}')
        d = prepare_mixture_dataset(
            cls_target,
            args['batch_size'],
            mixture_pct=cut,
            normalisation=args['norm_choice'])

        # perform #repetitions per 5% dataset mixture
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
            results[i,j] = balanced_accuracy_score(d['test_labels'], model.predict(d['test_data'](), steps=d['test_steps']).argmax(axis=1))
    print(results)
    np.save(join(repo_path, 'data/synthetic_influence_target_{cls_target}', results))

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
        choices=[0, 1, 2],
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
        choices=[0, 1, 2],
        default=2,
        help='Which normalisation to use. 0=None, 1=snv, 2=minmax',
        dest='norm_choice'
    )
    args = parser.parse_args()

    classify(**vars(args))
