from abc import ABC
from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

from storage.MetaInfo import ModelMetaInfo

KL_STEP = 1.0 / 5000.0
DROPOUT_PROB = 0.1


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, a latent semantic vector from random.normal"""

    def call(self, inputs, *args, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim), mean=0.0, stddev=1.0, dtype=tf.float32)
        return z_mean + tf.sqrt(tf.exp(z_log_var)) * epsilon


def create_encoder(vocab_dim: int, hidden_dim: int, latent_dim: int, dropout_prob: float = DROPOUT_PROB):
    """Creates a VDSH encoder: vocab_dim -> latent_dim"""
    inputs = tf.keras.Input(shape=(vocab_dim,))
    x = layers.Dense(hidden_dim, activation="relu", name="enc1")(inputs)
    x = layers.Dense(hidden_dim, activation="relu", name="enc2")(x)
    x = layers.Dropout(rate=dropout_prob)(x)

    z_mean = layers.Dense(latent_dim, activation="linear", name="to_mean")(x)
    z_log_var = layers.Dense(latent_dim, activation="sigmoid", name="to_logvar")(x)

    z = Sampling()([z_mean, z_log_var])

    encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    return encoder


def create_decoder(vocab_size: int, latent_dim: int):
    """Creates a VDSH decoder: latent_dim -> vocab_dim"""
    latent_inputs = tf.keras.Input(shape=(latent_dim,))

    prob_w = layers.Dense(vocab_size, activation="softmax")(latent_inputs)
    logprob_w = tf.math.log(prob_w)

    decoder = tf.keras.Model(latent_inputs, logprob_w, name="decoder")
    decoder.summary()

    return decoder


class VDSH(Model, ABC):
    def __init__(self, encoder, decoder, kl_step=KL_STEP, **kwargs):
        """Create a VDSH model by supplying encoder and decoder models"""
        super(VDSH, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.kl_weight = 0.0
        self.kl_step = kl_step

        self.meta: Optional[ModelMetaInfo] = None

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruct_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    @staticmethod
    def reconstruct_loss(logprob_words, inputs):
        return -tf.reduce_mean(tf.reduce_sum(logprob_words * inputs, axis=1))

    @staticmethod
    def kl_loss(z_mean, z_log_var):
        """Kullback–Leibler divergence"""
        loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        return loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            logprob_w = self.decoder(z)

            rl = self.reconstruct_loss(logprob_w, data)
            kld = self.kl_loss(z_mean, z_log_var)
            loss = rl + self.kl_weight * kld

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.total_loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(rl)
        self.kl_loss_tracker.update_state(kld * self.kl_weight)

        self.kl_weight = min(self.kl_weight + self.kl_step, 1.0)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def call(self, inputs, training=None, mask=None):
        """Creates a z_mean vector by encoding the input"""
        z_mean, z_log_var, z = self.encoder(inputs)
        return z_mean
