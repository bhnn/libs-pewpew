import numpy as np
from kerastuner.tuners import RandomSearch
from sklearn.utils import class_weight
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from utils import prepare_dataset, set_classification_targets


def build_model(params, id=0, num_classes=4, name='model'):
    model_name = name if name else f'model_{id}'
    inputs = Input(shape=(7810,))
    net = inputs

    dense_sizes = [64, 128, 256]
    reg = regularizers.l2
    reg_lambas = [1e-3, 1e-4, 1e-5, 1e-6]
    for layer in range(params.Int('num_layers', min_value=1, max_value=4, default=1)):
        net = Dense(
            params.Choice(f'dense_size_L{layer}', dense_sizes),
            activation='relu',
            kernel_regularizer=reg(0.0001)#params.Choice(f'dense_reg_lb_L{layer}', reg_lambas))
        )(net)
        net = Dropout(0.5)(net)
    net = Dense(
        params.Choice(f'dense_size_last', dense_sizes),
        activation='relu',
        kernel_regularizer=reg(0.0001)#params.Choice(f'dense_reg_lb_last', reg_lambas))
    )(net)
    net = Dense(num_classes, activation='softmax')(net)

    model = Model(inputs=inputs, outputs=net, name=model_name)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

batch_size = 64
# determine classification targets and parameters to construct datasets properly
num_classes, cls_target, cls_str = set_classification_targets(0)
train_data, test_data, train_labels, test_labels, epoch_steps, _ = prepare_dataset(2, cls_target, num_classes, batch_size)

# list of "class" names used for confusion matrices and validity testing. Not always classes, also subgroups or minerals
class_names = [i for i in range(num_classes)]
class_weights = class_weight.compute_class_weight('balanced', class_names, train_labels)

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=150,
    executions_per_trial=3,
    directory='/home/ben/Dropbox/uni/3_semester/ml/libs-pewpew/results',
    project_name='tuned_mlp'
)

tuner.search_space_summary(extended=True)
print('')

tuner.reload()
# tuner.search(
#     train_data,
#     steps_per_epoch=epoch_steps,
#     epochs=10,
#     class_weight=class_weights,
#     verbose=0,
#     validation_data=test_data
# )

tuner.results_summary()

print(tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values)