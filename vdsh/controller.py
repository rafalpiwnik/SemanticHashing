from vdsh.VDSH import create_encoder, create_decoder, VDSH


def create_model(vocab_dim: int, hidden_dim: int, latent_dim: int, dropout_prob: float):
    """Creates a VDSH model"""
    enc = create_encoder(vocab_dim, hidden_dim, latent_dim, dropout_prob)
    dec = create_decoder(vocab_dim, latent_dim)

    vdsh = VDSH(encoder=enc, decoder=dec)

    return vdsh
