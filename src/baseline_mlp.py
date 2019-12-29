import tensorflow as tf
import numpy as np

train_samples, train_labels, test_samples, test_labels = np.load('/home/ben/Desktop/ML/synthetic_data/final.npy', allow_pickle=True)

train_data = tf.data.Dataset.from_tensor_slices((train_samples, train_labels)).shuffle(10000).batch(32)
test_data  = tf.data.Dataset.from_tensor_slices((test_samples, test_labels)).batch(32)

