import copy

import numpy as np
import pytest

from vdsh.VDSH import *


@pytest.fixture
def vocab_size():
    return 100


@pytest.fixture
def hidden_dim():
    return 10


@pytest.fixture
def latent_dim():
    return 10


def test_vdsh_trains(vocab_size, hidden_dim, latent_dim):
    """Asserts that weights in the model undergo changes"""
    encoder = create_encoder(vocab_size, hidden_dim, latent_dim)
    decoder = create_decoder(vocab_size, latent_dim)
    model = VDSH(encoder, decoder)
    model.compile(optimizer="adam")

    x = np.ones(shape=(1, vocab_size))

    before = copy.deepcopy(model.trainable_variables)

    model.fit(x, batch_size=1, epochs=5)
    after: list[tf.Variable] = model.trainable_variables

    assert len(before) == len(after)

    are_diff = []
    for b, a in zip(before, after):
        print(b)
        print(a)
        diff: tf.Tensor = tf.math.reduce_any(b != a)
        are_diff.append(diff.numpy())

    print(are_diff)
    assert any(are_diff)
