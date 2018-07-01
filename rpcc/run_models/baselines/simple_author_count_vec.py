import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer

from rpcc import DATA_DIR


def clean(doc):
    return list(doc.split(''))


def clean_up_authors(doc):
    """
    Extract author names from a string

    :param authors: str. the authors fo the article
    :return: list of str. with the authors of the article, is applicable
    """
    if doc is None:
        return ''

    else:
        temp_authors = re.sub(r'\((.*?)\)', '', doc)
        temp_authors = re.sub(r'\((.*?,)', '', temp_authors)
        temp_authors = re.sub(r'\((.*?)', '', temp_authors)

        cleaned_authors = temp_authors.split(',')
        cleaned_authors = [a.strip() for a in cleaned_authors]

        return cleaned_authors

# read data
train_data = pd.read_csv(DATA_DIR + '/raw/train.csv')
train_data.columns = ['id', 'target']
test_data = pd.read_csv(DATA_DIR + '/raw/test.csv')
test_data = pd.DataFrame(data=test_data['Article'])
test_data.columns = ['id']
metadata = pd.read_csv(DATA_DIR + '/raw/node_information.csv')

print(train_data.columns)
print(test_data.columns)
print(metadata.columns)

train_data_all = pd.merge(train_data, metadata, on='id')
test_data_all = pd.merge(test_data, metadata, on='id')

lb = LabelBinarizer()
y_train_one_hot = lb.fit_transform(train_data_all['target'])

print(y_train_one_hot)
print(y_train_one_hot.shape)
print(train_data_all.columns)

v = CountVectorizer(min_df=5, max_df=100, stop_words='english', tokenizer=clean)

x = None

train_data_all['authors'] = train_data_all['authors'].apply(lambda x: x if x else ' ')
test_data_all['authors'] = test_data_all['authors'].apply(lambda x: x if x else ' ')

x_train = v.fit_transform(train_data_all['authors']).toarray()
x_test = v.transform(test_data_all['authors']).toarray()

print(x_train.shape)
print(x_test.shape)
