from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from rpcc.features_text import *
from rpcc.load_data import load_graph, load_train_ids, load_data
from rpcc.run_models import run_grid_search

if __name__ == "__main__":
    data = load_data()

    X_train = pd.DataFrame(data['abstracts'], columns=['abstracts'])
    y_train = pd.Series(data['labels'])

    # X_train_lemmatized = pd.DataFrame(LemmaExtractor(col_name='abstracts').fit_transform(X_train))

    vect_based_pipeline = Pipeline([('extract', TextColumnExtractor(column='abstracts')),
                                    ('contractions', ContractionsExpander()),
                                    ('vect', CountVectorizer()),
                                    ('tfidf', TfidfTransformer()),
                                    ('to_dense', DenseTransformer()),  # transforms sparse to dense
                                    ('scaling', StandardScaler()),
                                    # ('scaling', MinMaxScaler()),
                                    # ('pca', PCA()),
                                    # ('clf', SVC()),
                                    # ('clf', MultinomialNB())
                                    ('clf', LogisticRegression())
                                    # ('clf', KNeighborsClassifier())
                                    # ('clf', GradientBoostingClassifier())
                                    # ('clf', RandomForestClassifier())
                                    ])

    params = {
        'vect__min_df': (0.01, 0.05),
        'vect__max_features': (None, 1000, 2500, 5000),
        'vect__stop_words': ('english',),
        'vect__binary': (True, False),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams, bigrams, trigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l2',),
        # 'tfidf__smooth_idf': (True, False),  # do not use
        # 'tfidf__sublinear_tf': (True, False),  # do not use
        'clf__penalty': ('l1', 'l2'),  # Logistic
        # 'clf__kernel': ('rbf', 'linear'),  # SVM
        # 'clf__gamma': (0.1, 0.01, 0.001, 0.0001),  # SVM
        # 'clf__p': (1, 2),  # 1: mahnatan, 2: eucledian # k-NN
        # 'clf__n_neighbors': (3, 4, 5, 6, 7, 8),  # k-NN
        # 'clf__learning_rate': (0.1, 0.01, 0.001),  # Gradient Boosting
        # 'clf__n_estimators': (100, 300, 600),  # Gradient Boosting, Random Forest
        # 'clf__alpha': (0.5, 1.0),  # MultinomialNB
        # 'clf__max_depth': [10, 50, 100, None],  # Random Forest
    }

    grid_results = run_grid_search(X=X_train,
                                   y=y_train,
                                   pipeline=vect_based_pipeline,
                                   parameters=params,
                                   scoring='accuracy')

    # setting the best classifier.
    # best_clf = LogisticRegression(C=0.1, penalty='l1')

    # # setting the final pipeline.
    # final_pipeline = Pipeline([
    #     ('features', FeatureUnion(transformer_list=[
    #         ('vect_based_feat', vect_based_features),
    #         ('user_based_feat', user_based_features)])),
    #     ('scaling', StandardScaler()),
    #     ('clf', best_clf)])# setting the best classifier.
    # best_clf = LogisticRegression(C=0.1, penalty='l1')
    #
    # # setting the final pipeline.
    # final_pipeline = Pipeline([
    #     ('features', FeatureUnion(transformer_list=[
    #         ('vect_based_feat', vect_based_features),
    #         ('user_based_feat', user_based_features)])),
    #     ('scaling', StandardScaler()),
    #     ('clf', best_clf)])
