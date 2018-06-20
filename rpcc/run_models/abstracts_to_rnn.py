from sklearn.preprocessing import LabelBinarizer

from rpcc.create_features import TextFeaturesExtractor
from rpcc.load_data import DataLoader

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

    print(x_train_padded)
    # tfe_obj.text_to_padded_sequences(texts=x_text,
    #                                  tokenizer=tokenizer,
    #                                  max_length=,max_length)
