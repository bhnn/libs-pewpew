import argparse

from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras import Model
from tensorflow.keras.layers import Average, Input
from tensorflow.keras.utils import plot_model

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
        args['batch_size'])

    print('\n\tTask: Classify «{}» using «{}»\n'.format(cls_str, d['data_str']))
    print_dataset_info(d)

    # build and train
    inputs = Input(shape=(7810,))
    models = [build_model(i, d['num_classes'], inputs=inputs) for i in range(args['num_models'])]

    # combine outputs of all models
    y = Average()([m.outputs[0] for m in models])
    model = Model(inputs, outputs=y, name='multiple')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if allow_print:
        model.summary()
        print('')
        plot_model(model, to_file='img/multiple_mlp.png')

    model.fit(
        d['train_data'],
        steps_per_epoch=d['train_steps'],
        epochs=args['epochs'],
        verbose=1,
        class_weight=d['class_weights'])

    # evaluation model
    print('Evaluate ...')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.evaluate(d['eval_data'], steps=d['test_steps'], verbose=1)

    # predict on testset and calculate classification report and confusion matrix for diagnosis
    print('Test ...')
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
        '-d', '--dataset',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help='Which dataset(s) to use. 0=synthetic, 1=hh_12, 2=hh_all',
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
        '-m', '--models',
        type=int,
        default=4,
        help='How many models the ensemble should use',
        dest='num_models'
    )
    args = parser.parse_args()

    repeat_and_collate(classify, **vars(args))
