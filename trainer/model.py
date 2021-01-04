"""
This script is inspired by https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/tf-keras/trainer/model.py.

The input_fn() function is used to take our datasets (which are numpy.array objects) and make them tf.data.Dataset(s).

The create_keras_model() function is a nice and simple wrapper to create a Keras model of our choice.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def input_fn(features, labels, shuffle, num_epochs, batch_size):
    """Generates an input function to be used for model training.

    Args:
      features: numpy array of features used for training or inference
      labels: numpy array of labels for each example
      shuffle: boolean for whether to shuffle the data or not (set True for
        training, False for evaluation)
      num_epochs: number of epochs to provide the data for
      batch_size: batch size for training

    Returns:
      A tf.data.Dataset that can provide data to the Keras model for training or
        evaluation
    """
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    return dataset


def create_keras_model(vocab_size, embedding_dim, max_document_len):
    """
    Creates Keras Model for Text Classification.

    Arguments
    ---------
    vocab_size (int) : the number of words in our vocabulary.
    embedding_dim (int) : the length of vector we should embed word tokens to.
    max_document_len (int) : the length of the longest document in our data.

    Returns
    -------
    tf.keras.Sequential : a compiled Keras model ready to be trained.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_document_len,
                mask_zero=True,
            ),
            tf.keras.layers.LSTM(units=64, activation=tf.nn.tanh),
            tf.keras.layers.Dense(units=2, activation=tf.nn.sigmoid),
        ]
    )

    # Compile Keras model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
