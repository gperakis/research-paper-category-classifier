from sklearn.preprocessing import LabelBinarizer

from rpcc.create_features import TextFeaturesExtractor
from rpcc.load_data import DataLoader
from rpcc.models import AbstractEmbedding
from keras.models import load_model
import pandas as pd
import os
from rpcc import DATA_DIR

if __name__ == "__main__":
    dl_obj = DataLoader()
    dl_obj.run_data_preparation()

    x_test = dl_obj.x_test['title']

    lb = LabelBinarizer()

    tfe_obj = TextFeaturesExtractor(input_data=x_test)

    meta = tfe_obj.pre_process_text(texts=x_test)

    x_test_padded = meta['x']
    int2word = meta['int2word']
    word2int = meta['word2int']
    max_length = meta['max_length']
    tokenizer = meta['tokenizer']

    padded_sequences = tfe_obj.text_to_padded_sequences(texts=x_test,
                                                        tokenizer=tokenizer,
                                                        max_length=18)

    loaded_model = load_model('title_rnn.h5')

    preds = pd.DataFrame(loaded_model.predict(padded_sequences))
    preds.columns = sorted(dl_obj.targets)
    preds.index = dl_obj.x_test['Article']
    preds.reset_index(inplace=True)

    outfile_path = os.path.join(DATA_DIR, 'predictions_titles.csv')
    preds.to_csv(outfile_path, sep=',', index=False)


