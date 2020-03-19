import argparse

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model

from utils import (build_model, build_model_concat, prepare_dataset,
                   set_classification_targets, print_dataset_info, diagnose_output)


def classify(**args):
    """
    Main method that prepares dataset, builds model, executes training and displays results.
    
    :param args: keyword arguments passed from cli parser
    """
    batch_size = 64
    # determine classification targets and parameters to construct datasets properly
    cls_target, cls_str = set_classification_targets(args['cls_choice'])
    d = prepare_dataset(
        0, # any synthetic
        cls_target,
        batch_size)

    print('\n\tTask: Classify «{}» using «{}»\n'.format(cls_str, d['data_str']))
    print_dataset_info(d)

    model = build_model(1, d['num_classes'], name='concat_mlp', new_input=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
        1, # any handheld
        cls_target,
        batch_size)

    # build model for handheld data, concatenates the output of the last pre-classification layer of the synthetic network
    concat_model = build_model_concat(2, d['num_classes'], concat_model=model)
    concat_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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

    diagnose_output(d['test_labels'], pred.argmax(axis=1), d['classes_trans'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--classification',
        type=int,
        default=2,
        help='Which dataset(s) to use. 0=synthetic, 1=hh_6, 2=hh_12, 3=hh_all',
        dest='cls_choice'
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=5,
        help='How many epochs to train for',
        dest='epochs'
    )
    args = parser.parse_args()

    classify(**vars(args))
