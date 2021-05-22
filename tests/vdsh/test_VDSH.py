from vdsh.controller import create_vdsh


def test_run():
    vdsh = create_vdsh(vocab_dim=5000, hidden_dim=500, latent_dim=32, kl_step=(1/5000.0), dropout_prob=0.1)