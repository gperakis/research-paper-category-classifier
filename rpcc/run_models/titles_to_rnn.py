from sklearn.preprocessing import LabelBinarizer

from rpcc.create_features import TextFeaturesExtractor
from rpcc.load_data import DataLoader
from rpcc.models import AbstractEmbedding
from keras.models import load_model
import pandas as pd


if __name__ == "__main__":
    dl_obj = DataLoader()
    dl_obj.run_data_preparation()

    x_train = dl_obj.x_train['title']
    y_train = dl_obj.y_train

    lb = LabelBinarizer()

    y_train_one_hot = lb.fit_transform(y_train)
    tfe_obj = TextFeaturesExtractor(input_data=x_train)

    meta = tfe_obj.pre_process_text(texts=x_train)

    x_train_padded = meta['x']
    int2word = meta['int2word']
    word2int = meta['word2int']
    max_length = meta['max_length']
    tokenizer = meta['tokenizer']

    ab_emb_obj = AbstractEmbedding(emb_size=200,
                                   voc_size=len(int2word),
                                   max_sequence_length=max_length)
    #
    # ab_emb_obj.build_model(dropout=0.7,
    #                        rnn_size=100)
    #
    # ab_emb_obj.fit(X=x_train_padded,
    #                y=y_train_one_hot,
    #                epochs=150,
    #                val_size=0.0,
    #                bs=128,
    #                lr=0.001)
    # ab_emb_obj.model.save('title_rnn.h5')

    ab_emb_obj.model = load_model('title_rnn.h5')
    ab_emb_obj.fit(X=x_train_padded,
                   y=y_train_one_hot,
                   epochs=500,
                   val_size=0.0,
                   bs=128,
                   lr=0.001)
    print('saving model')
    ab_emb_obj.model.save('title_rnn.h5')

    print('running prediction')
    x_test = dl_obj.x_test['title']
    x_test_padded = tfe_obj.text_to_padded_sequences(texts=x_test,
                                                     tokenizer=tokenizer,
                                                     max_length=max_length)

    preds = pd.DataFrame(ab_emb_obj.model.predict(x=x_test_padded))
    preds.columns = sorted(dl_obj.targets)
    preds.index = dl_obj.x_test['Article']
    preds.reset_index(inplace=True)
    print('preds are ready and about to be saved.')

    preds.to_csv('title_preds.csv', encoding='utf-8', index=False)