import numpy as np
import tensorflow as tf
from h5sparse import h5sparse

from addressing.metrics import run_recall_test
from controllers.usersetup import setup_homedir, load_config
from preprocess import DocumentVectorizer
from vdsh import create_vdsh


def train_mock(train, train_target, test, test_target):
    VOCAB_SIZE = 10000

    vectorizer = DocumentVectorizer(VOCAB_SIZE)
    # paths = get_paths("datasets/mini_newsgroups")
    # X_sparse, words = vectorizer.fit_transform(paths)

    vdsh = create_vdsh(VOCAB_SIZE, hidden_dim=1000, latent_dim=32, kl_step=(1 / 5000.0), dropout_prob=0.1)

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)

    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    vdsh.compile(optimizer=opt)

    vdsh.fit(train, batch_size=100, epochs=10)

    train_pred = vdsh.predict(train)
    test_pred = vdsh.predict(test)

    run_recall_test(train_pred, train_target, test_pred, test_target)


if __name__ == "__main__":
    print("Starting")

    setup_homedir(overwrite=False)
    settings = load_config()

    print(settings)

    # fetch_20ng(vocab_size=10000)

    path = settings["model"]["data_home"]
    with h5sparse.File(f"{path}/20ng/data.hdf5", "r") as hf:
        train = np.array(hf["train"])
        train_labels = np.array(hf["train_labels"])
        test = np.array(hf["test"])
        test_labels = np.array(hf["test_labels"])

        train_mock(train, train_labels, test, test_labels)
