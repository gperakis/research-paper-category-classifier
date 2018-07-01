from __future__ import print_function

import csv
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from rpcc import DATA_DIR
from rpcc.load_data import load_data
from rpcc.run_models import run_grid_search


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key."""

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.key]


class TitleAbstractExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        out = dict()

        out['title'] = list(df['title'])
        out['abstract'] = list(df['abstract'])

        return out


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'length': len(text),
                 'num_sentences': text.count('.')}
                for text in posts]


if __name__ == "__main__":
    train_data = load_data()
    test_data = load_data(csv_file='test.csv')
    test_ids = test_data['csv_df']['Article']

    pipeline = Pipeline([
        # Extract the title & abstract
        ('title_abstract', TitleAbstractExtractor()),

        # Use FeatureUnion to combine the features from title and abstract
        ('union', FeatureUnion(
            transformer_list=[

                # Pipeline for pulling features from the post's title line
                ('title', Pipeline([
                    ('selector', ItemSelector(key='title')),
                    ('tfidf', TfidfVectorizer(min_df=50, stop_words='english')),
                ])),

                # Pipeline for standard bag-of-words model for abstract
                ('abstract_bow', Pipeline([
                    ('selector', ItemSelector(key='abstract')),
                    ('tfidf', TfidfVectorizer(stop_words='english')),
                    ('best', TruncatedSVD(n_components=50)),
                ])),

                # Pipeline for pulling ad hoc features from post's abstract
                ('abstract_stats', Pipeline([
                    ('selector', ItemSelector(key='abstract')),
                    ('stats', TextStats()),  # returns a list of dicts
                    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ])),

            ],

            # weight components in FeatureUnion
            transformer_weights={
                'title': 0.8,
                'abstract_bow': 1.0,
                'abstract_stats': 0.5,
            },
        )),
        # Use a Logistic classifier on the combined features
        ('clf', XGBClassifier()),
    ])

    params = {
        'union__title__tfidf__ngram_range': [(1, 1), ],
        'union__abstract_bow__tfidf__ngram_range': [(1, 1), ],
        'union__abstract_bow__best__n_components': [50, 100, 150],
        'clf__penalty': ('l1', 'l2')  # Logistic
    }

    grid_search_obj = run_grid_search(X=train_data['csv_df'],
                                      y=train_data['labels'],
                                      pipeline=pipeline,
                                      parameters=params,
                                      scoring='accuracy')

    y_pred = grid_search_obj.predict_proba(test_data['csv_df'])

    # Write predictions to a file

    with open(os.path.join(DATA_DIR, 'sample_submission_bow.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        lst = grid_search_obj.classes_.tolist()
        lst.insert(0, "Article")
        writer.writerow(lst)
        for i, test_id in enumerate(test_ids):
            lst = y_pred[i, :].tolist()
            lst.insert(0, test_id)
            writer.writerow(lst)
