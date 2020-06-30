import argparse
from os import makedirs
from os.path import join

import matplotlib.pyplot as plt
import seaborn as sn
from numpy import max, mean, save, zeros
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from utils import (build_model, diagnose_output, prepare_dataset,
                   print_dataset_info, repeat_and_collate,
                   set_classification_targets)


def classify(**args):
    """
    Main method that prepares dataset, builds model, executes training and displays results.
    
    :param args: keyword arguments passed from cli parser
    """
    # only allow print-outs if execution has no repetitions
    allow_print = args['repetitions'] == 1
    # determine classification targets and parameters to construct datasets properly
    cls_target, cls_str = set_classification_targets(args['cls_choice'])
    d = prepare_dataset(
        args['dataset_choice'],
        cls_target,
        args['batch_size'],
        args['norm_choice'],
        mp_heatmap=True)

    print('\n\tTask: Classify «{}» using «{}»\n'.format(cls_str, d['data_str']))
    print_dataset_info(d)

    model = build_model(0, d['num_classes'], name='64shot_mlp', new_input=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if allow_print:
        model.summary()
        print('')

    # train and evaluate
    model.fit(
        x=d['train_data'],
        steps_per_epoch=d['train_steps'],
        epochs=args['epochs'],
        verbose=1,
        class_weight=d['class_weights'])

    model.evaluate(d['eval_data'], steps=d['test_steps'], verbose=1)

    # predict on testset and calculate classification report and confusion matrix for diagnosis
    pred = model.predict(d['test_data'], steps=d['test_steps'], verbose=1)
    # instead of argmax, reduce list to only on-target predictions to see how accurate the model judged each shot
    target_preds = [pred[i][l] for i,l in enumerate(d['test_labels'])]
    pred = pred.argmax(axis=1)

    # empty ndarray with max measurepoints +1 space for 8x8 grids
    acc_heatmap = zeros((max(d['heatmap_tm'], axis=0)[0] + 1, 8, 8))

    # use heatmap transition matrix to build 8x8 grid for each measure point
    # necessary because sometimes shots are missing in between or at the end of measure points
    for i, single_pred in enumerate(target_preds):
        acc_heatmap[d['heatmap_tm'][i][0], d['heatmap_tm'][i][1], d['heatmap_tm'][i][2]] = single_pred

    # info for titles and save directory
    target_str = 'Classes' if cls_target == 0 else ('Subgroups' if cls_target == 1 else 'Minerals')
    dest_dir = d['dataset_name'] + f'_c{cls_target}_e' + str(args['epochs']) + '_64heatmap'
    dest_path = join('results', dest_dir)
    makedirs(dest_path, exist_ok=True)

    # create heatmap image out of 8x8 grid for each measure point
    for i,m in tqdm(enumerate(acc_heatmap), desc='heatmap_plt', total=len(acc_heatmap)):
        fig = plt.figure(figsize = (10,7))
        sn.heatmap(m, annot=True, fmt='.2f')
        plt.gca().tick_params(axis='y', rotation=45)
        plt.title(f'Per shot accuracy of a single measure point (No. {i}), Target: {target_str}')
        plt.savefig(join(dest_path, f'mp_{i:03}.pdf'), bbox_inches='tight')
        plt.close(fig) # needs to be closed, otherwise 100 image panels will pop up next time plt.show() is called

    # average over all results
    mean_heatmap = mean(acc_heatmap, axis=0)

    plt.figure(figsize = (10,7))
    sn.heatmap(mean_heatmap, annot=True, fmt='.2f')
    plt.gca().tick_params(axis='y', rotation=45)
    plt.title(f'Per shot accuracy, mean over all measure points, Target: {target_str}')
    dataset_name = d['dataset_name']
    plt.savefig(join(dest_path, f'{dataset_name}_c{cls_target}.pdf'), bbox_inches='tight')
    plt.show()

    save(join(dest_path, f'{dataset_name}_c{cls_target}.npy'), acc_heatmap)

    return balanced_accuracy_score(d['test_labels'], pred)


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
        '-d', '--dataset',
        type=int,
        choices=[1, 2],
        default=1,
        help='Which dataset(s) to use. 1=hh_12, 2=hh_all',
        dest='dataset_choice'
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
        default=10,
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

    repeat_and_collate(classify, **vars(args))
