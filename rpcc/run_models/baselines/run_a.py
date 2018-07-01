from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from rpcc.features_text import *
from rpcc.load_data import restore_data_loader
from rpcc.run_models import run_grid_search

if __name__ == "__main__":
    dl_obj = restore_data_loader()

    x_train_val = dl_obj.x_train_validation
    y_train_val = dl_obj.y_train_validation

    x_train = dl_obj.x_train
    y_train = dl_obj.y_train

    x_val = dl_obj.x_val
    y_val = dl_obj.y_val

    print(x_train_val.keys())

    X_train = pd.DataFrame(data['abstracts'], columns=['abstracts'])
    y_train = pd.Series(data['labels'])

    final_pipeline = Pipeline(
        [
            ('features', FeatureUnion(transformer_list=[
                ('text_length', TextLengthExtractor(col_name='abstracts')),
                ('avg_token_length',
                 WordLengthMetricsExtractor(col_name='abstracts', metric='avg', split_type='simple')),
                ('std_token_length',
                 WordLengthMetricsExtractor(col_name='abstracts', metric='std', split_type='simple')),
                ('n_tokens', NumberOfTokensCalculator(col_name='abstracts')),
            ])),
            ('scaling', StandardScaler()),
            ('clf', LogisticRegression())])

    params = {
        'features__contains_uppercase__how': ['bool', 'count'],
        'clf__penalty': ('l1', 'l2'),  # Logistic
        # 'clf__kernel': ('rbf', 'linear'),  # SVM
        # 'clf__gamma': (0.1, 0.01, 0.001, 0.0001),  # SVM
        # 'clf__p': (1, 2),  # 1: mahnatan, 2: eucledian # k-NN
        # 'clf__n_neighbors': (3, 4, 5, 6, 7, 8),  # k-NN
        # 'clf__learning_rate': (0.1, 0.01, 0.001),  # Gradient Boosting
        # 'clf__n_estimators': (100, 300, 600),  # Gradient Boosting, Random Forest
        # 'clf__alpha': (0.1, 0.5, 1.0),  # MultinomialNB
        # 'clf__fit_prior': (True, False),  # MultinomialNB
        # 'clf__max_depth': [10, 50, 100, None],  # Random Forest
    }

    grid_results = run_grid_search(X=X_train,
                                   y=y_train,
                                   pipeline=final_pipeline,
                                   parameters=params,
                                   scoring='accuracy')
