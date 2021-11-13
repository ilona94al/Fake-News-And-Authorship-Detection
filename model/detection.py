import os

from model.fake_bert_model import FakeBERTModel

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


class Detection():
    def __init__(self, input, model_name,model_type):
        self.model=FakeBERTModel()
        #os.chdir("../TRAINED_MODELS/" + model_type)
        self.model.load_model("TRAINED_MODELS/"+model_type+"/"+model_name)
        #os.chdir("../../gui_controllers")
        input_texts=self.get_preproccessed_text(input,max_text_len=self.model.config['max_seq_len'])

        predicted_prob = self.model.predict(tf.constant(input_texts))
        predicted = np.argmax(predicted_prob, -1)

        self.fake_numb = np.count_nonzero(predicted)
        self.real_numb = np.count_nonzero(predicted == 0)
        self.all = predicted.shape

        # self.real_percent= 100 * self.real_numb / all
        # self.fake_percent= 100 * self.fake_numb / all

        X_axis = np.arange(predicted_prob.shape[0]) + 1

        plt.bar(X_axis - 0.2, predicted_prob[:, 0], 0.4, label='Real')
        plt.bar(X_axis + 0.2, predicted_prob[:, 1], 0.4, label='Fake')

        plt.xlabel("Book parts")
        plt.ylabel("Prob")
        plt.title("---")
        plt.legend()
        plt.savefig()




    def get_preproccessed_text(self, input,max_text_len):
        input_texts = []
        from model.preprocessing import text_preprocessing
        preprocessed_input = text_preprocessing(input)
        from model.preprocessing import separate_text_to_blocks
        input_blocks = separate_text_to_blocks(preprocessed_input,max_text_len)
        input_texts.extend((block for block in input_blocks))
        return input_texts

