import argparse

import pydotplus
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from tqdm import tqdm

from utils import (diagnose_output, prepare_dataset, print_dataset_info,
                   set_classification_targets)


def classify(**args):
    """
    Main method that prepares dataset, builds model, executes training and displays results.
    
    :param args: keyword arguments passed from cli parser
    """
    batch_size = 64
    # determine classification targets and parameters to construct datasets properly
    cls_target, cls_str = set_classification_targets(args['cls_choice'])
    d = prepare_dataset(
        args['dataset_choice'],
        cls_target,
        batch_size,
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
    test_data = [sample for batch in tqdm(d['test_data'], total=d['test_steps'], desc='prep_test') for sample in batch[0]]
    pred = model.predict(test_data)
    del test_data

    # visualise decision tree, from datacamp.com/community/tutorials/decision-tree-classification-python
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('img/decision_tree.pdf')

    diagnose_output(d['test_labels'], pred, d['classes_trans'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        type=int,
        default=1,
        help='Which dataset(s) to use. 0=synthetic, 1=hh_6, 2=hh_12, 3=hh_all',
        dest='dataset_choice'
    )
    parser.add_argument(
        '-c', '--classification',
        type=int,
        default=2,
        help='Which classification target to pursue. 0=classes, 1=subgroups, 2=minerals',
        dest='cls_choice'
    )
    args = parser.parse_args()

    classify(**vars(args))
