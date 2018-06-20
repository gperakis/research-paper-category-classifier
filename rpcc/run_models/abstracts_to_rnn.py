from sklearn.preprocessing import LabelBinarizer

from rpcc.create_features import TextFeaturesExtractor
from rpcc.load_data import DataLoader
from rpcc.models import AbstractEmbedding

if __name__ == "__main__":
    dl_obj = DataLoader()
    dl_obj.run_data_preparation()

    x_train = dl_obj.x_train['abstract']
    y_train = dl_obj.y_train

    lb = LabelBinarizer()

    y_train_one_hot = lb.fit_transform(y_train)

    print(y_train_one_hot.shape)

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

    ab_emb_obj.build_model(dropout=0.3,
                           rnn_size=50)

    ab_emb_obj.fit(X=x_train_padded,
                   y=y_train_one_hot,
                   epochs=30,
                   val_size=0.2)

    ab_emb_obj.model.save('abstract_rnn.h5')

    # tfe_obj.text_to_padded_sequences(texts=x_text,
    #                                  tokenizer=tokenizer,
    #                                  max_length=,max_length)
