from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf


class DistilRoBERTaEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(DistilRoBERTaEncoder, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
        self.roberta_layer = TFRobertaModel.from_pretrained("distilroberta-base")

    def call(self, input_ids, training):
        roberta_output = self.roberta_layer(input_ids, training=training)
        return roberta_output.last_hidden_state
