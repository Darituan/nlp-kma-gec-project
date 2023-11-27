import tensorflow as tf
from transformer.distilroberta_encoder import DistilRoBERTaEncoder
from transformer.decoder import Decoder


class Transformer(tf.keras.Model):

    def __init__(self, num_layers=None, d_model=None, num_heads=None, dff=None,
                 target_vocab_size=None, pe_target=None, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = DistilRoBERTaEncoder(d_model=d_model)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)


    def call(self, input_ids, target_ids, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(input_ids, training)

        dec_output, attention_weights = self.decoder(
            target_ids, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
