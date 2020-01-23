import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, Input,
                                     MaxPooling1D)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import Precision, Recall

from utils import normalise_minmax, transform_labels

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

inputs = Input(shape=(7810,1))
net = Conv1D(filters=16, kernel_size=3, activation='relu')(inputs)
net = Conv1D(filters=32, kernel_size=3, activation='relu')(net)
net = Dropout(0.5)(net)
net = MaxPooling1D(pool_size=4)(net)
net = Flatten()(net)
net = Dense(150, activation='relu')(net)
net = Dense(12, activation='softmax')(net)

model = keras.Model(inputs=inputs, outputs=net)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy', Precision(), Recall()])
model.fit(train_data, steps_per_epoch=250, epochs=2, verbose=1, use_multiprocessing=True)
model.evaluate(test_data, verbose=1)

# pred = model.predict(test_data, use_multiprocessing=True).argmax(axis=1)
# mineral_list = np.load('data/synthetic_minerals.npy', allow_pickle=True)
# class_names = [f'{row[0]}_{row[1]}' for row in mineral_list]
# # print(classification_report(y_true=test_labels[:,2].astype(int), y_pred=pred, labels=class_names))
# matrix = confusion_matrix(y_true=test_labels, y_pred=pred, labels=class_names)
