## Context

The purpose of this repo is to train a text classification model (using a recurrent neural network) in Keras and Google Cloud Platform.

I use data from https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences.

This collection includes three datasets:

1. Amazon Reviews
2. Yelp Reviews
3. IMDB Reviews

The datasets contain reviews (our predictor variable) and sentiment (our target variable).

For example:

`["This film was fantastic. You HAVE to watch it!", 1]`

`["Can't believe I watched this crap. If only you could give negative stars.", 0]`

Sentiment is labelled `0` if the review was negative. Sentiment is labelled `1` if the review was positive.

I focus on the dataset containing Amazon reviews `amazon_cells_labelled.txt` and achieve an accuracy of 81% :rocket:.

Note. The purpose this repo is not to achieve 100% accuracy. It's to showcase how to create a model in Keras and on GCP.

## Keras

All relevant files to train a model in Keras are found in [#1](https://github.com/jmc-bbk/text-classification-example/pull/1).

I follow a classic Data Science workflow of training a model on my local machine using Jupyter Notebook.

We have two important files:

1. `extract.ipynb` - this loads our data from a local source, converts it to a `Pandas.DataFrame`, and saves as a `.pkl` file.
2. `modelling.ipynb` - this opens our `.pkl` file, does some preprocessing, and then trains a Keras model.

The trained model is found in `models/lstm_model.h5`.

## Google Cloud Platform

All relevant files to train a custom Keras model on GCP are found in [#2](https://github.com/jmc-bbk/text-classification-example/pull/2).

There are three important files:

1. `model.py` - this converts datasets saved as `np.array` objects into `tf.data.Dataset` objects, which are required for GCP. It also compiles a Keras model (exactly what we did in `modelling.ipynb`).
2. `utils.py` - this is simply a collection of wrapper functions to download our data from GCP, preprocess it, and return it as `np.array` objects ready to be used by `model.py`.
3. `task.py` - this pulls everything together. It parses arguments from the command line to configure our GCP job. It then uses functions from `utils.py` to load our data and `models.py` to compile our model, before training and evaluating the model on our data.

## Notes

I highly recommend using the linked PRs [#1](https://github.com/jmc-bbk/text-classification-example/pull/1) and [#2](https://github.com/jmc-bbk/text-classification-example/pull/2) to understand what files are required for Keras and Google Cloud Platform.

In [#1](https://github.com/jmc-bbk/text-classification-example/pull/1) I train a model, use cross-validation, and evaluate on an unseen test dataset.

In [#2](https://github.com/jmc-bbk/text-classification-example/pull/2) I train a model and use cross-validation. You would have to take extra steps to test this on unseen test data.
