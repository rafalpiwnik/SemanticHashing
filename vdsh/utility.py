import os

import numpy as np
import tensorflow as tf

import storage
from controllers.usersetup import load_config
from storage import datasets, DocumentVectorizer
from storage.MetaInfo import ModelMetaInfo
from vdsh.VDSH import create_encoder, create_decoder, VDSH


def create_vdsh(vocab_dim: int, hidden_dim: int, latent_dim: int, kl_step: float, dropout_prob: float,
                name: str = "default"):
    """Creates a VDSH model and adds relevant meta info"""
    enc = create_encoder(vocab_dim, hidden_dim, latent_dim, dropout_prob)
    dec = create_decoder(vocab_dim, latent_dim)

    vdsh = VDSH(encoder=enc, decoder=dec, kl_step=kl_step)

    mi = ModelMetaInfo(name=name,
                       vocab_size=vocab_dim,
                       hidden_dim=hidden_dim,
                       latent_dim=latent_dim,
                       kl_step=kl_step,
                       dropout_prob=dropout_prob)

    vdsh.meta = mi

    return vdsh


def create_mock_model(vocab_dim: int, hidden_dim: int, latent_dim: int, kl_step: float, dropout_prob: float,
                      name: str = "default"):
    """Creates and saves a mock VDSH model by filling in the meta info only"""
    mi = ModelMetaInfo(name=name,
                       vocab_size=vocab_dim,
                       hidden_dim=hidden_dim,
                       latent_dim=latent_dim,
                       kl_step=kl_step,
                       dropout_prob=dropout_prob)

    model_dest = f'{load_config()["model"]["model_home"]}/{name}'

    try:
        os.mkdir(model_dest)
    except FileExistsError:
        pass

    mi.dump(model_dest)

    return mi


def train_model(model: VDSH, batch: int, num_epochs: int, dataset_name: str):
    """Fits model with train data of a given qualified dataset and saves it under its qualified name to models
    The model must be compiled

    Parameters
    ----------
    model : VDSH
        A compiled VDSH model
    batch : int
        Batch size to use in training
    num_epochs : int
        Number of epochs
    dataset_name : str
        Qualified dataset name of a dataset folder located at data_home/dataset_name


    Returns
    -------
    VDSH
        Fitted model with updated meta info

    """
    train = datasets.extract_train(dataset_name)  # Gets dataset by name from data_home

    model.fit(train, batch_size=batch, epochs=num_epochs)  # Fits with train only
    model.predict(train[0:, ])

    model.meta.flag_fit(dataset_name)  # Model is set as fitted
    dump_model(model)  # Model is dumped, alongside with meta info and vectorizer is available

    return model


def dump_model(model: VDSH):
    """Dumps the model and the vectorizer at data_home/fit_dataset to model_home/model.meta.name

    Intended to use after fitting a model with a given dataset
    Saved model cannot be refitted directly but has to be copied

    Parameters
    ----------
    model : VDSH
        Model with meta info

    Returns
    -------
    None
    """
    config = load_config()
    model_home = config["model"]["model_home"]

    # Infer export model name and dataset name from meta info
    mi = model.meta
    model_name = mi.name
    dataset_name = mi.dataset_name

    model_dest = f"{model_home}/{model_name}"

    try:
        os.mkdir(model_dest)
    except FileExistsError:
        pass

    # Running predict to set up weights
    vocab_size = mi.vocab_size
    model.predict(np.zeros(shape=(1, vocab_size)))

    model.save(model_dest)
    mi.dump(model_dest)

    if dataset_name:
        datasets.copy_vectorizer(dataset_name, model_name)


def load_model(model_name: str) -> tuple[VDSH, DocumentVectorizer]:
    """Loads the model from model_home/model_name

    If the model itself exists it is returned itself, compiled and ready to use

    If the model doesn't exist but there is meta info file, then the model is created ad hoc
    and returned as well

    Parameters
    ----------
    model_name : str
        A qualified model name

    Returns
    -------
    tuple[VDSH, DocumentVectorizer]
        Retrieved model and the vectorizer if present, else None
    """
    config = load_config()
    model_home = config["model"]["model_home"]

    mi = ModelMetaInfo.from_file(f"{model_home}/{model_name}")

    try:
        model = tf.keras.models.load_model(f"{model_home}/{model_name}")
    except OSError:
        print("Model not found. Creating model...")
        model = create_vdsh(mi.vocab_size,
                            mi.hidden_dim,
                            mi.latent_dim,
                            mi.kl_step,
                            mi.dropout_prob,
                            mi.name)

    # Push meta info to model
    model.meta = mi

    try:
        vec = storage.load_vectorizer(f"{model_home}/{model_name}")
    except (FileNotFoundError, IOError):
        print("Vectorizer not found")
        vec = None

    print("Model loaded:")
    print(model.meta.info)

    return model, vec
