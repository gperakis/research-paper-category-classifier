from shutil import rmtree
from tempfile import mkdtemp

import pandas as pd
from sklearn.externals.joblib import Memory
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rpcc.features_text import *
from rpcc.load_data import load_data
from rpcc.run_models import run_grid_search

if __name__ == "__main__":
    data = load_data()

    print(data.keys())
    X_train = pd.DataFrame(data['abstracts'], columns=['abstracts'])
    y_train = pd.Series(data['labels'])

    we_obj = GloveWordEmbedding()

    pre_loaded_we = {
        50: we_obj.get_word_embeddings(dimension=50),
        100: we_obj.get_word_embeddings(dimension=100),
        # 200: we_obj.get_word_embeddings(dimension=200),
        # 300: we_obj.get_word_embeddings(dimension=300)
    }

    # Create a temporary folder to store the transformers of the pipeline
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=10)

    final_pipeline = Pipeline(
        [
            ('embedding_feat', SentenceEmbeddingExtractor(col_name='abstracts',
                                                          word_embeddings_dict=pre_loaded_we)),
            ('scaling', StandardScaler()),
            # ('scaling', MinMaxScaler()),
            # ('pca', PCA()),
            # ('clf', SVC()),
            # ('clf', MultinomialNB())
            ('clf', LogisticRegression())
            # ('clf', KNeighborsClassifier())
            # ('clf', GradientBoostingClassifier())
            # ('clf', RandomForestClassifier())
        ],
        memory=memory)

    params = {
        'embedding_feat__embedding_type': ['tfidf', 'tf'],  # embedding
        'embedding_feat__embedding_dimensions': [50, ],  # embedding  100, 200, 300
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
                                   pipeline=final_pipeline,
                                   parameters=params,
                                   scoring='accuracy')

    # Delete the temporary cache before exiting
    rmtree(cachedir)
