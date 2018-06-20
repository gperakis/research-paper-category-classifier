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

    predicted_classes = loaded_model.predict(padded_sequences)

    print(predicted_classes.shape)

    header = ["Acta", "Adv.Theor.Math.Phys.", "Annals", "Class.Quant.Grav.", "Commun.Math.Phys.",
              "Eur.Phys.J.", "Fortsch.Phys.", "Int.", "Int.J.Mod.Phys.", "Int.J.Theor.Phys.", "J.Geom.Phys.",
              "J.Math.Phys.", "J.Phys.", "JHEP", "Lett.Math.Phys.", "Mod.", "Mod.Phys.Lett.", "Nucl.",
              "Nucl.Phys.", "Nucl.Phys.Proc.Suppl.", "Nuovo", "Phys.", "Phys.Lett.", "Phys.Rev.",
              "Phys.Rev.Lett.", "Prog.Theor.Phys.", "Theor.Math.Phys.", "Z.Phys."]

    x = pd.DataFrame(data=x_test.index.values, columns=['Article'])
    prediction_df = pd.DataFrame(data=predicted_classes, columns=header)

    result = pd.concat([x, prediction_df], axis=1)

    print(result.head())

    outfile_path = os.path.join(DATA_DIR, 'predictions_titles.csv')
    result.to_csv(outfile_path, sep='\t', index=False)
