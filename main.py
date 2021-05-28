import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model

import vdsh.utility
from addressing import file_search
from addressing.metrics import run_recall_test
from controllers.controller import create_user_dataset
from storage import DocumentVectorizer
from storage.datasets import extract_train
from vdsh import create_vdsh


def train_mock(train, train_target, test, test_target):
    VOCAB_SIZE = 100

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
    # setup_homedir(overwrite=False)

    VOCAB_SIZE = 10000
    HIDDEN_DIM = 1000
    LATENT_DIM = 32

    # Train schema to date
    """
    create_user_dataset(root_dir="C:\\Users\\rafal\\Desktop\\20_newsgroups", vocab_size=VOCAB_SIZE, name="20ng_user")

    model = vdsh.utility.create_vdsh(VOCAB_SIZE, HIDDEN_DIM, LATENT_DIM, 1 / 5000.0, 0.1, name="20ng_user")
    model.compile(optimizer="adam")

    vdsh.utility.train_model(model, 100, 15, "20ng_user")
    """

    model = vdsh.utility.create_vdsh(VOCAB_SIZE, HIDDEN_DIM, LATENT_DIM, 1 / 5000.0, 0.1, name="20ng_user")

    # model, vec = vdsh.utility.load_model("20ng_user")
    # predict = model.predict(np.zeros(shape=(1, 10000)))
    # print(predict)

    # vdsh.utility.train_model(model, 100, 10, dataset_name="20ng_user")

    # Working file search
    # file_search.search("20ng_user", "C:\\Users\\rafal\\Desktop\\20_newsgroups",
    #                    "C:\\Users\\rafal\\Desktop\\20_newsgroups\\talk.politics.guns\\53294")
