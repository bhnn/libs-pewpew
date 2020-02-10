import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight

from utils import build_model, prepare_dataset, set_classification_targets


def classify(**args):
    batch_size = 64
    # determine classification targets and parameters to construct datasets properly
    num_classes, cls_target, cls_str = set_classification_targets(args['cls_choice'])
    train_data, test_data, train_labels, test_labels, epoch_steps, data_str = prepare_dataset(args['dataset_choice'], cls_target, num_classes, batch_size, return_tf_dataset=False)

    # list of "class" names used for confusion matrices and validity testing. Not always classes, also subgroups or minerals
    class_names = [i for i in range(num_classes)]
    class_weights = class_weight.compute_class_weight('balanced', class_names, train_labels)
    print('class weights:', class_weights)
    
    print(f'\n\tTask: Classify «{cls_str}» using «{data_str}» in DecisionTreeClassifier\n')

    model = DecisionTreeClassifier(class_weight='balanced')

    # model = build_model(0, num_classes, name='baseline_mlp', new_input=True)
    # model.compile(optimizer=Adam(learning_rate=0.001, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    # print('')
    # plot_model(model, to_file='img/decision_tree.png')


    # train and evaluate
    model.fit(train_data, train_labels)

    # predict on testset and calculate classification report and confusion matrix for diagnosis
    pred = model.predict(test_data)
    print('Balanced accuracy score: ', balanced_accuracy_score(y_true=test_labels, y_pred=pred))
    print(classification_report(y_true=test_labels, y_pred=pred, labels=class_names))

    from sklearn.externals.six import StringIO
    from sklearn.tree import export_graphviz
    import pydotplus

    # visualise decision tree, from datacamp.com/community/tutorials/decision-tree-classification-python
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('img/decision_tree.pdf')

    # normalised confusion matrix
    matrix = confusion_matrix(y_true=test_labels, y_pred=pred, labels=class_names)
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
