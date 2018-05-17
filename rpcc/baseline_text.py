import csv
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from rpcc import DATA_DIR

# Load data about each article in a dataframe
df = pd.read_csv(os.path.join(DATA_DIR, "node_information.csv"))
print(df.head())

# Read training data
train_ids = list()
y_train = list()
with open(os.path.join(DATA_DIR, 'train.csv'), 'r') as f:
    next(f)
    for line in f:
        t = line.split(',')
        train_ids.append(t[0])
        y_train.append(t[1][:-1])

n_train = len(train_ids)
unique = np.unique(y_train)
print("\nNumber of classes: ", unique.size)

# Extract the abstract of each training article from the dataframe
train_abstracts = list()
for i in train_ids:
    train_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])

# Create the training matrix. Each row corresponds to an article 
# and each column to a word present in at least 2 webpages and at
# most 50 articles. The value of each entry in a row is equal to 
# the frequency of that word in the corresponding article	
vec = CountVectorizer(decode_error='ignore', min_df=5, max_df=200, stop_words='english')
X_train = vec.fit_transform(train_abstracts)

# Read test data
test_ids = list()
with open(os.path.join(DATA_DIR, 'test.csv'), 'r') as f:
    next(f)
    for line in f:
        test_ids.append(line[:-2])

# Extract the abstract of each test article from the dataframe
n_test = len(test_ids)
test_abstracts = list()
for i in test_ids:
    test_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])

# Create the test matrix following the same approach as in the case of the training matrix
X_test = vec.transform(test_abstracts)

print("\nTrain matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)

# Use logistic regression to classify the articles of the test set
text_clf = LogisticRegression(class_weight=None)
text_clf.fit(X_train, y_train)
y_pred = text_clf.predict_proba(X_test)

# Write predictions to a file
with open(os.path.join(DATA_DIR, 'sample_submission_text.csv'), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = text_clf.classes_.tolist()
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i, test_id in enumerate(test_ids):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)
