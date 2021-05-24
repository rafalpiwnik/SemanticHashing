import os

import tensorflow as tf

from controllers.usersetup import load_config
from preprocess import fetch_dataset
from vdsh.VDSH import create_encoder, create_decoder, VDSH


def create_vdsh(vocab_dim: int, hidden_dim: int, latent_dim: int, kl_step: float, dropout_prob: float):
    """Creates a VDSH model"""
    enc = create_encoder(vocab_dim, hidden_dim, latent_dim, dropout_prob)
    dec = create_decoder(vocab_dim, latent_dim)

    vdsh = VDSH(encoder=enc, decoder=dec, kl_step=kl_step)

    return vdsh


def fit_processed(model: VDSH, batch: int, num_epochs: int, dataset_name: str):
    train = fetch_dataset.extract_train(dataset_name)

    model.fit(train, batch_size=batch, epochs=num_epochs)
    model.predict(train[0:,])

    dump_model(model, model.name)
    fetch_dataset.copy_vectorizer(dataset_name, model.name)

    return model


def dump_model(model: VDSH, target_dir_name: str):
    try:
        model_home = load_config()["model"]["model_home"]
        dest = f"{model_home}/{target_dir_name}"

        try:
            os.mkdir(dest)
        except FileExistsError:
            pass

        model.save(dest)

    except (KeyError, IOError):
        print("Couldn't read config.json file")
