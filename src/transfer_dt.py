import argparse

import pydotplus
from sklearn.externals.six import StringIO
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from tqdm import tqdm

from utils import (diagnose_output, prepare_dataset, print_dataset_info,
                   repeat_and_collate, set_classification_targets)


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
        0, # any synthetic
        cls_target,
        args['batch_size'],
        train_shuffle_repeat=False,
        categorical_labels=False)

    print('\n\tTask: Classify «{}» using «{}» with DecisionTreeClassifier\n'.format(cls_str, d['data_str']))
    print_dataset_info(d)

    model = DecisionTreeClassifier(class_weight='balanced')

    # empty train data generator into list, then train. Careful with RAM
    train_data = [sample for batch in tqdm(d['train_data'], total=d['train_steps'], desc='prep_train') for sample in batch[0]]
    model.fit(train_data, d['train_labels'])
    del train_data

    # predict on testset and calculate classification report and confusion matrix for diagnosis
    d = prepare_dataset(
        2, # any handheld
        cls_target,
        args['batch_size'],
        train_shuffle_repeat=False,
        categorical_labels=False)
    test_data = [sample for batch in tqdm(d['test_data'], total=d['test_steps'], desc='prep_test') for sample in batch[0]]
    print_dataset_info(d)
    pred = model.predict(test_data)
    del test_data

    if allow_print:
        # visualise decision tree, from datacamp.com/community/tutorials/decision-tree-classification-python
        dot_data = StringIO()
        export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf('img/decision_tree.pdf')

        diagnose_output(d['test_labels'], pred, d['classes_trans'])

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
        '-c', '--classification',
        type=int,
        default=2,
        help='Which classification target to pursue. 0=classes, 1=subgroups, 2=minerals',
        dest='cls_choice'
    )
    args = parser.parse_args()

    repeat_and_collate(classify, **vars(args))
