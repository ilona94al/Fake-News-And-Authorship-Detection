import os

from model.fake_bert_model import FakeBERTModel

import tensorflow as tf

import numpy as np




class Detection():
    def __init__(self, input, model_name, model_type):
        self.model = FakeBERTModel()

        #   loads the relevant model according to name and type
        self.model.load_model("TRAINED_MODELS/" + model_type + "/" + model_name)

        #   input preprocessing (separate input to chunks in size 128 tokens)
        input_texts = self.get_preprocessed_text(input, max_text_len=self.model.config['max_seq_len'])
        self.number_of_chunks= len(input_texts) + 1

        self.probabilities, self.predictions = self.get_prediction(input_texts)

    @staticmethod
    def get_preprocessed_text(input, max_text_len):
        input_texts = []
        from model.preprocessing import text_preprocessing
        preprocessed_input = text_preprocessing(input)
        from model.preprocessing import separate_text_to_blocks
        input_blocks = separate_text_to_blocks(preprocessed_input, max_text_len)
        input_texts.extend((block for block in input_blocks))
        return input_texts

    def get_prediction(self, input):
        #   get a prediction for input chunks
        predicted_probs = self.model.predict(tf.constant(input))
        #   returned 2d array like this:
        #   probabilities for each chunk to belong for each label
        #         ----     Real(0)    |     Fake(1)
        #        chunk1     0.09      |      0.91
        #        chunk2     0.78      |      0.22
        #         ...       ...       |      ...

        #   put the output from model to array in this format:
        #       | chunk1 | chunk2 |   ...
        #       |   1    |    0   |   ...
        predictions_array = np.argmax(predicted_probs, -1)

        return predicted_probs, predictions_array
