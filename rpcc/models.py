from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
import numpy as np
from pprint import pprint
from keras import backend as K

from keras import backend as K


class MyModel:
    def __init__(self,
                 emb_size: int,
                 voc_size: int,
                 max_sequence_length: int):
        """

        """
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.max_sequence_length = max_sequence_length

        self.model: Model = None
        self.history: list = None

    def build_model(self):
        return None

    def fit(self, X, y, epochs: int = 10):
        """

        :param X:
        :param y:
        :param model:
        :param epochs:
        :return:
        """
        self.history = self.model.fit(X,
                                      y,
                                      epochs=epochs,
                                      batch_size=128)
        return self.history

    def predict(self):
        pass

    def get_last_layer_values_before_activation(self, x_input):
        """

        :param x_input:
        :return:
        """

        inp = self.model.input  # input placeholder
        outputs = [self.model.layers[-2].output]  # all layer outputs

        functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
        layer_outs = functor([x_input, 1.])

        return layer_outs[0]


class AbstractEmbedding(MyModel):
    def __init__(self, emb_size: int, voc_size: int, max_sequence_length: int):
        super().__init__(emb_size, voc_size, max_sequence_length)

    def build_model(self):
        """

        :param max_sequence_length:
        :param emb_size:
        :param voc_size:
        :return:
        """

        # define model
        text_input = Input(shape=(self.max_sequence_length,), dtype='int32', name='text')
        embedded_text = layers.Embedding(self.voc_size, self.emb_size)(text_input)
        encoded_text1 = layers.LSTM(32, return_sequences=True)(embedded_text)
        encoded_text2 = layers.LSTM(32)(encoded_text1)

        category = layers.Dense(2, activation='sigmoid')(encoded_text2)

        model = Model(text_input, [category])

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])

        print(model.summary())

        self.model = model

class TitleEmbedding(MyModel):
    def __init__(self, emb_size: int, voc_size: int, max_sequence_length: int):
        super().__init__(emb_size, voc_size, max_sequence_length)

    def build_model(self):
        pass


class KCoreEmbedding(MyModel):
    def __init__(self, emb_size: int, voc_size: int, max_sequence_length: int):
        super().__init__(emb_size, voc_size, max_sequence_length)

    def build_model(self):
        pass


class AuthorEmbedding(MyModel):
    def __init__(self, emb_size: int, voc_size: int, max_sequence_length: int):
        super().__init__(emb_size, voc_size, max_sequence_length)

    def build_model(self):
        pass


if __name__ == '__main__':
    text_vocabulary_size = 10000
    num_samples = 100
    max_length = 100

    X = np.random.randint(1,
                          text_vocabulary_size,
                          size=(num_samples, max_length))

    y = to_categorical(np.random.randint(low=0, high=2, size=num_samples))

    obj = AbstractEmbedding(emb_size=10,
                            voc_size=text_vocabulary_size,
                            max_sequence_length=max_length)

    obj.build_model()

    obj.fit(X, y, 15)

    asdf = obj.get_last_layer_values_before_activation(x_input=X)

    pprint(asdf)

    pprint(asdf.shape)
