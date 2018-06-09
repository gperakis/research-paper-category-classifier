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
    def __init__(self):
        """

        """
        self.model: Model = None
        self.history: list = None

    def build_model(self):
        pass

    def fit(self, X, y, model, epochs: int = 10):
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
    def __init__(self):
        super().__init__()

    @staticmethod
    def build_model(max_sequence_length: int,
                    emb_size: int,
                    voc_size: int):
        """

        :param max_sequence_length:
        :param emb_size:
        :param voc_size:
        :return:
        """

        # define model
        text_input = Input(shape=(max_sequence_length,), dtype='int32', name='text')
        embedded_text = layers.Embedding(voc_size, emb_size)(text_input)
        encoded_text1 = layers.LSTM(32, return_sequences=True)(embedded_text)
        encoded_text2 = layers.LSTM(32)(encoded_text1)

        category = layers.Dense(2, activation='sigmoid')(encoded_text2)

        model = Model(text_input, [category])

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])

        print(model.summary())

        out = {'model': model,
               'last_lstm': encoded_text2}

        return out


class TitleEmbedding(MyModel):
    def __init__(self):
        super().__init__()

    def build_model(self):
        pass


class KCoreEmbedding(MyModel):
    def __init__(self):
        super().__init__()

    def build_model(self):
        pass


class AuthorEmbedding(MyModel):
    def __init__(self):
        super().__init__()

    def build_model(self):
        pass


if __name__ == '__main__':
    import numpy as np

    text_vocabulary_size = 10000
    num_samples = 100
    max_length = 100

    X = np.random.randint(1,
                          text_vocabulary_size,
                          size=(num_samples, max_length))

    y = to_categorical(np.random.randint(low=0, high=2, size=num_samples))

    obj = AbstractEmbedding()
    meta = obj.build_model(max_sequence_length=max_length,
                           emb_size=10,
                           voc_size=text_vocabulary_size)

    # trained_model = obj.fit(X, y, meta['model'])
