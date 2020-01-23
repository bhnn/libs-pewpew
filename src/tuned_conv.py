import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from kerastuner import HyperParameters
from kerastuner.tuners import RandomSearch

from utils import normalise_minmax, transform_labels


def build_model(params, outputs=12):
    inputs = layers.Input(shape=(7810,1))
    net = inputs
    for num_layers in range(params.Int('num_layers', min_value=1, max_value=4)):
        net = layers.Conv1D(filters=params.Int(f'filters_{num_layers}_1', min_value=8, max_value=24, step=8), 
                            kernel_size=params.Int(f'kernels_{num_layers}_1', min_value=5, max_value=25, step=10),
                            activation='relu')(net)
        net = layers.Conv1D(filters=params.Int(f'filters_{num_layers}_2', min_value=32, max_value=64, step=8),
                            kernel_size=params.Int(f'kernels_{num_layers}_2', min_value=5, max_value=25, step=10),
                            activation='relu')(net)
        net = layers.Dropout(rate=0.5)(net)
        net = layers.MaxPooling1D(pool_size=params.Int('pooling', min_value=4, max_value=16, step=4))(net)

    net = layers.Flatten()(net)
    net = layers.Dense(params.Int('dense', min_value=50, max_value=250, step=50), activation='relu')(net)
    net = layers.Dense(outputs, activation='softmax')(net)

    model = keras.Model(inputs=inputs, outputs=net)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    return model

train_samples, train_labels, test_samples, test_labels = np.load('/home/ben/Desktop/ML/synthetic_data/final.npy', allow_pickle=True)
# normalise each sample with its own np.max
train_samples_norm = normalise_minmax(train_samples)
test_samples_norm = normalise_minmax(test_samples)
# reshape for (batch, timesteps, channels) for Conv1D operation
train_samples_norm = np.reshape(train_samples_norm, (*train_samples_norm.shape, 1))
test_samples_norm = np.reshape(test_samples_norm, (*test_samples_norm.shape, 1))
# create onehot vectors out of labels
train_labels = transform_labels(train_labels)
test_labels = transform_labels(test_labels)
train_labels_onehot = keras.utils.to_categorical(train_labels[:,2].astype(int), 12)
test_labels_onehot = keras.utils.to_categorical(test_labels[:,2].astype(int), 12)
# use Dataset as input pipeline
train_data = tf.data.Dataset.from_tensor_slices((train_samples_norm, train_labels_onehot)).shuffle(10000).batch(32, drop_remainder=True).repeat(-1)
test_data  = tf.data.Dataset.from_tensor_slices((test_samples_norm, test_labels_onehot)).batch(32)

params = HyperParameters()
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='/home/ben/Dropbox/uni/3_semester/ml/libs-pewpew/results',
    project_name='tuned_conv'
)
tuner.search_space_summary()
tuner.search(
    train_data,
    steps_per_epoch=250,
    epochs=2,
    validation_data=test_data
)

tuner.results_summary()