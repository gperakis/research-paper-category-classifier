from keras import layers
from keras import models

from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
import numpy as np
from pprint import pprint

from keras import backend as K


class Model:
    def __init__(self):
        pass

    def fit(self, X, y, model):
        history = model.fit(X, y,
                            epochs=10,
                            batch_size=128)

        last_layer_tensor = model.layers[2].output

        return history

    def predict(self):
        pass


class AbstractEmbedding(Model):
    def __init__(self):
        super().__init__()

    @staticmethod
    def build_model(max_sequence_length: int, emb_size: int = 10):
        # sequence_input = layers.Input(shape=(max_sequence_length,), dtype='int32')

        # model = models.Sequential()
        # model.add(layers.LSTM(5, return_sequences=True, input_shape=(max_sequence_length, 1)))
        # model.add(layers.LSTM(3))
        # model.add(layers.Dense(1, activation='sigmoid'))
        #
        # model.compile(optimizer='rmsprop',
        #               loss='binary_crossentropy',
        #               metrics=['acc'])
        #
        # print("Model Fitting - Bidirectional LSTM")
        # print(model.summary())

        # define model
        inputs1 = Input(shape=(max_sequence_length,), dtype='int32')
        embeddings = layers.Embedding(None, emb_size)(inputs1)
        lstm1 = LSTM(5, return_sequences=True)(embeddings)
        lstm2 = LSTM(3)(lstm1)
        dense = layers.Dense(1, activation='sigmoid')(lstm2)

        model = Model(inputs=inputs1, outputs=[lstm2, dense])

        return model


class TitleEmbedding(Model):
    def __init__(self):
        super().__init__()

    def build_model(self):
        pass


class KCoreEmbedding(Model):
    def __init__(self):
        super().__init__()

    def build_model(self):
        pass


class AuthorEmbedding(Model):
    def __init__(self):
        super().__init__()

    def build_model(self):
        pass


if __name__ == '__main__':
    X = np.array([
        [13, 28, 18, 7, 9, 5],
        [29, 44, 38, 15, 26, 22],
        [27, 40, 31, 29, 32, 1]]).reshape((3, 6, 1))

    pprint(X)

    y = np.array(['1', '0', '0'])

    obj = AbstractEmbedding()
    model = obj.build_model(6)
    obj.fit(X, y, model)
