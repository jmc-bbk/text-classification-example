"""
This script is inspired by https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/tf-keras/trainer/task.py

It's actually incredibly simple. There are two components:

get_args() : parses arguments from our CLI which can be used elsewhere in the script.

train_and_evaluate() : loads data using our utils.py script, trains and evaluates a model on that data using our model.py script.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from . import model
from . import utils

import tensorflow as tf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir",
        type=str,
        required=True,
        help="local or GCS location for writing checkpoints and exporting " "models",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="number of times to go through the data, default=20",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="number of records to read during each training step, default=64",
    )
    parser.add_argument(
        "--embedding_dim",
        default=64,
        type=int,
        help="size of vector to embed input word tokens to.",
    )
    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
        default="INFO",
    )
    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate(args):

    object_dict = utils.load_data()

    X_train, X_val, y_train, y_val = (
        object_dict["X_train"],
        object_dict["X_val"],
        object_dict["y_train"],
        object_dict["y_val"],
    )

    max_document_len = object_dict["max_document_len"]
    vocab_size = object_dict["vocab_size"]

    num_train_examples = X_train.shape[0]
    num_val_examples = X_val.shape[0]

    keras_model = model.create_keras_model(
        vocab_size=vocab_size,
        max_document_len=max_document_len,
        embedding_dim=args.embedding_dim,
    )

    training_dataset = model.input_fn(
        features=X_train,
        labels=y_train,
        shuffle=False,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )

    validation_dataset = model.input_fn(
        features=X_val,
        labels=y_val,
        shuffle=False,
        num_epochs=args.num_epochs,
        batch_size=num_val_examples,
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        os.path.join(args.job_dir, "keras_tensorboard"), histogram_freq=1
    )

    keras_model.fit(
        training_dataset,
        steps_per_epoch=int(num_train_examples / args.batch_size),
        epochs=args.num_epochs,
        validation_data=validation_dataset,
        validation_steps=1,
        verbose=1,
        callbacks=[tensorboard_cb],
    )

    export_path = os.path.join(args.job_dir, "keras_export")
    keras_model.save(export_path)
    print("Model exported to: {}".format(export_path))


if __name__ == "__main__":
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
