import tensorflow as tf

from preprocess import DocumentVectorizer, get_paths
from vdsh import create_vdsh

if __name__ == "__main__":
    VOCAB_SIZE = 10000

    vectorizer = DocumentVectorizer(VOCAB_SIZE)
    paths = get_paths("datasets/mini_newsgroups")
    X_sparse, words = vectorizer.fit_transform(paths)
    X = X_sparse.toarray()

    vdsh = create_vdsh(VOCAB_SIZE, hidden_dim=1000, latent_dim=128, kl_step=(1 / 5000.0), dropout_prob=0.1)

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)

    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    vdsh.compile(optimizer=opt)

    vdsh.fit(X, batch_size=100, epochs=25)
