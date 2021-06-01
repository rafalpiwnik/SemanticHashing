import tensorflow as tf

from addressing.metrics import run_recall_test
from vdsh import create_vdsh


# Sample test procedure
def train_mock(train, train_target, test, test_target):
    VOCAB_SIZE = 100

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
    print("Run MainWindow to use the gui")
