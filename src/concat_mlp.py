import argparse

from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model

from utils import (build_model, build_model_concat, diagnose_output,
                   prepare_dataset, print_dataset_info, repeat_and_collate,
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
        0, # any synthetic
        cls_target,
        args['batch_size'])

    print('\n\tTask: Classify «{}» using «{}»\n'.format(cls_str, d['data_str']))
    print_dataset_info(d)

    model = build_model(1, d['num_classes'], name='concat_mlp', new_input=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if allow_print:
        model.summary()
        plot_model(model, to_file='img/concat_mlp.png')

    # train and evaluate
    model.fit(
        d['train_data'],
        steps_per_epoch=d['train_steps'],
        epochs=args['epochs'],
        verbose=1,
        class_weight=d['class_weights'])
    model.evaluate(d['eval_data'], steps=d['test_steps'], verbose=1)

    del d

    # load handheld dataset for evaluation
    d = prepare_dataset(
        2, # any handheld
        cls_target,
        args['batch_size'])
    print_dataset_info(d)

    # build model for handheld data, concatenates the output of the last pre-classification layer of the synthetic network
    concat_model = build_model_concat(2, d['num_classes'], concat_model=model)
    concat_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if allow_print:
        concat_model.summary()
        plot_model(concat_model, to_file='img/concat_mlp.png')

    concat_model.fit(
        d['train_data'],
        steps_per_epoch=d['train_steps'],
        epochs=args['epochs'],
        verbose=1,
        class_weight=d['class_weights'])

    # predict on test set and calculate classification report and confusion matrix for diagnosis
    pred = model.predict(d['test_data'], steps=d['test_steps'])

    if allow_print:
        diagnose_output(d['test_labels'], pred.argmax(axis=1), d['classes_trans'])
    
    return balanced_accuracy_score(d['test_labels'], pred.argmax(axis=1)) 

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
        default=10,
        help='How many epochs to train for',
        dest='epochs'
    )
    args = parser.parse_args()

    repeat_and_collate(classify, **vars(args))
