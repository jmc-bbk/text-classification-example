"""
This script is inspired by https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/tf-keras/trainer/util.py

The purpose is to create a utility function load_data() that will:

1. download data from a source (e.g. GCP Cloud Storage)
2. preprocess data
3. return X_train, X_val, y_train, y_val

We store the data in a temporary file (see download()) that's of the form /tmp/our_dir_name/our_file_name.

My load_data() function isn't the cleanest but it's functional.

The scripts in GCP tutorials return X_train, X_val, y_train, y_val as a tuple.

I return two extra python objects (vocab_size, max_document_len) and do so in a dict. It really doesn't matter.

You can return whatever objects you need for the task.py script to run.

One thing missing from this script: I should really save the fitted Tokenizer back to my GCP bucket. This would
be needed later to preprocess new texts for predictions.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.cloud import storage
from sklearn.model_selection import train_test_split

import os
import tempfile
import random

import numpy as np
import pandas as pd
import tensorflow as tf

# Set the download directory for our data.
BUCKET_NAME = "YOUR BUCKET NAME"  # I have hidden my bucket name.
BLOB = "YOUR BLOB PATH"  # I have hidden my blob path.


# Set the local directory to store our data.
DATA_DIR = os.path.join(tempfile.gettempdir(), "text_data")
FILE_NAME = "amazon_cells_labelled.txt"

# Define a global variable to set random seeds and states.
RANDOM_STATE = 7


def _download_blob(bucket_name, source_blob_name, destination_file_name):
    """
    Downloads a blob (object) from google cloud bucket.

    Arguments
    ---------
    bucket_name (str) : name of bucket to download from.
    source_blob_name (str) : path to blob (object) to download.
    destination_file_name (str) : path to local storage.
    """

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def download(data_dir):
    """
    Wrapper for _download_blob. Downloads blob if it doesn't already
    exist in the temporary file location.

    Arguments
    ---------
    data_dir (str) : path to temporary file location.
    """

    file_path = os.path.join(data_dir, FILE_NAME)

    if not tf.io.gfile.exists(file_path):
        tf.io.gfile.makedirs(data_dir)
        _download_blob(BUCKET_NAME, BLOB, file_path)

    return file_path


def file_to_object(path):
    """
    Returns a list of data samples [document, sentiment] from our raw data.

    Document is referred to as X (this is our predictor variable).
    Sentiment is reffered to  as y (this is our target variable).

    Arguments
    ---------
    path (str) : a path to our raw data to open.

    Returns
    -------
    out (list) : a list of lists. Internal list is of structure [document (str), sentiment (int)]
    """

    out = list()

    with open(path, "r") as file:
        for line in file:
            split_line = (
                line.split()
            )  # split the raw line of text from the .txt file into a list.
            X = " ".join(
                split_line[:-1]
            )  # take the document (sentence) assign it to X as a str.
            y = int(split_line[-1])  # take the sentiment assign it to y as an int.
            out.append([X, y])  # append [document, sentiment] to the outer list.

    return out


def create_dataframe(_object):
    """
    Returns a single Pandas.DataFrame that contains reviews from one of the raw data sources.

    Arguments
    ---------
    file (str) : a path to the file to open.

    Returns
    -------
    pd.DataFrame : a dataframe that has two columns [document, sentiment].
    """

    return pd.DataFrame(
        _object, columns=["document", "sentiment"]
    )  # use pandas to make the data a DataFrame.


# Probably not the cleanest wrapper.
def y_preprocess(y):
    return tf.keras.utils.to_categorical(y)


def fit_tokenizer(X):
    """
    Returns a Tokenizer that's been fitted on the full vocab.

    Arguments
    ---------
    X (np.array) : an array of the full dataset (train+validation).

    Returns
    -------
    tokenizer (tf...Tokenizer) : a tokenizer fitted on X.
    """

    tokenizer = tf.python.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X)
    return tokenizer


def get_max_document_len(X):
    """
    Returns the maximum length of a document in X. Necessary for padding inputs.
    Also required to specify the input_dim for the Embedding Layer of our model.

    Arguments
    ---------
    X (np.array) : an array of the full dataset (train+validation).

    Returns
    -------
    max_document_len (int) : maximum length of document in train+validation data.
    """

    return max([len(doc.split()) for doc in X])


def get_vocab_size(tokenizer):
    """
    Returns the vocab size of X. Necessary for the Emebdding Layer of our model.

    Arguments
    ---------
    X (np.array) : an array of the full dataset (train+validation).

    Returns
    -------
    vocab_size (int) : the number of words in our vocabulary.
    """

    return len(tokenizer.word_index) + 1


def tokenize_sequences(X, tokenizer, max_document_len):
    """
    Returns tokenized and padded sequences for documents in X.

    Arguments
    ---------
    X (np.array) : an array of the full dataset (train+validation).
    tokenizer (tf...Tokenizer) : a tokenizer fitted on X.
    max_document_len (int) : maximum length of document in train+validation data.

    Returns
    -------
    sequences (np.array) : a processed X dataframe suitable for a keras model.
    """

    tokens = tokenizer.texts_to_sequences(X)
    return tf.python.keras.preprocessing.sequence.pad_sequences(
        tokens, maxlen=max_document_len, padding="post"
    )


def split_train_eval(X, y):
    """
    Wrapper for sk-learn train_test_split function. Splits data into
    train and test sample. Shuffles data and leaves 10% for test set.

    Arguments
    ---------
    X (np.array) : an array of the full dataset (train+validation).
    y (np.array) : an array of the target variable (train+validation).

    Returns
    -------
    X_train (np.array) : 90% of the data for training, shuffled.
    X_val (np.array) : 10% of the data for validation, shuffled.
    y_train (np.array) : 90% of the target variable for training, shuffled.
    y_val (np.array) : 10% of the target variable for validation, shuffled.
    """

    random.seed(RANDOM_STATE)
    return train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=RANDOM_STATE
    )


def load_data():
    """
    Function to be called by task.py to load data into our training script.
    Loads data, preprocesses data, returns processed and split dataframes.

    Returns
    -------
    dict : contains X_train, X_val, y_train, y_val, max_document_len, vocab_size.
    """

    file_path = download(DATA_DIR)
    dataframe = create_dataframe(file_to_object(file_path))

    X, y = dataframe["document"].values, dataframe["sentiment"].values

    tokenizer = fit_tokenizer(X)
    max_document_len = get_max_document_len(X)
    vocab_size = get_vocab_size(tokenizer)

    _X = tokenize_sequences(X, tokenizer, max_document_len)
    _y = y_preprocess(y)

    X_train, X_val, y_train, y_val = split_train_eval(_X, _y)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "max_document_len": max_document_len,
        "vocab_size": vocab_size,
    }
