import pandas as pd
from rpcc import DATA_DIR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelBinarizer
from rpcc.models import FeedForward
import os
import csv
from sklearn.linear_model import LogisticRegression


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

print(test_data_all)

lb = LabelBinarizer()
y_train_one_hot = lb.fit_transform(train_data_all['target'])

print(y_train_one_hot)
print(y_train_one_hot.shape)
print(train_data_all.columns)

v = TfidfVectorizer(min_df=5, max_df=100, stop_words='english')

x_train = v.fit_transform(train_data_all['abstract']).toarray()
x_test = v.transform(test_data_all['abstract']).toarray()

print(x_train.shape)
print(x_test.shape)

# svd = TruncatedSVD()
# X_transformed = svd.fit_transform(x)
# print(X_transformed.shape)

# model = FeedForward(100, 100, x_train.shape[1])
# model.build_model()
# model.fit(x_train, y_train_one_hot, epochs=5)
# preds = model.model.predict(x_test)

# Use logistic regression to classify the articles of the test set
text_clf = LogisticRegression(class_weight=None)
text_clf.fit(x_train, train_data_all['target'])
y_pred = text_clf.predict_proba(x_test)

print(y_pred)

# preds = pd.DataFrame(y_pred)
# preds.columns = sorted(dl_obj.targets)
# preds.index = dl_obj.x_test['Article']
# preds.reset_index(inplace=True)
#
# outfile_path = os.path.join(DATA_DIR, 'predictions_titles.csv')
# preds.to_csv(outfile_path, sep=',', index=False)

with open(os.path.join(DATA_DIR, 'predictions_tfidf.csv'), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = text_clf.classes_.tolist()
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i, test_id in enumerate(test_data['id']):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)

