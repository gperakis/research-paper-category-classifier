from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from rpcc import features_graph
from rpcc.load_data import restore_data_loader, DataLoader
from rpcc.run_models import run_grid_search
from rpcc.features_graph import create_combined_paper_authors_graph_features
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
if __name__ == "__main__":
    dl_obj = restore_data_loader()

    cites_graph_features = features_graph.create_cites_graph_features(load=True, save=False)
    authors_2_papers_graph_features = create_combined_paper_authors_graph_features(load=True, save=False)

    all_features = cites_graph_features.merge(authors_2_papers_graph_features, on='Article', how='left')

    train_ids = dl_obj.train_ids
    validation_ids = dl_obj.validation_ids

    y_train = dl_obj.y_train
    y_test = dl_obj.y_test

    # x_train = cites_graph_features[cites_graph_features['Article'].isin(train_ids)].set_index('Article')
    # x_validation = cites_graph_features[cites_graph_features['Article'].isin(validation_ids)].set_index('Article')
    #
    graph_based_pipe = Pipeline([
        ('scaling', StandardScaler()),
        # ('scaling', MinMaxScaler()),
        ('pca', PCA()),
        ('clf', SVC()),
        # ('clf', MultinomialNB())
        # ('clf', OneVsRestClassifier(LogisticRegression()))
        # ('clf', RandomForestClassifier())
        # ('clf', GradientBoostingClassifier())
        # ('clf', RandomForestClassifier())
    ])

    params = {
        # 'clf__penalty': ('l1', 'l2'),  # Logistic
        'pca__n_components': (5, 10, 15, 20, None),

        # 'clf__C': (0.1, 0.01, 0.001),
        # 'clf__n_neighbors': (3, 5, 7, 9, 10, 15, 20, 50, 100),
        # 'clf__leaf_size': (10, 20, 30, 50, 100),
        # 'clf__p': (2, 3, 5),

        'clf__kernel': ('rbf', 'linear'),  # SVM
        'clf__gamma': (0.1, 0.01, 0.001, 0.0001),  # SVM
        # 'clf__p': (1, 2),  # 1: mahnatan, 2: eucledian # k-NN
        # 'clf__n_neighbors': (3, 4, 5, 6, 7, 8),  # k-NN
        # 'clf__learning_rate': (0.1, 0.01, 0.001),  # Gradient Boosting
        # 'clf__n_estimators': (100, 300, 600),  # Gradient Boosting, Random Forest
        # 'clf__alpha': (0.5, 1.0),  # MultinomialNB
        # 'clf__max_depth': [2, 5, None],  # Random Forest
    }

    x_train_validation = all_features[all_features['Article'].isin(train_ids + validation_ids)].set_index('Article')

    grid_results = run_grid_search(X=x_train_validation,
                                   y=dl_obj.y_train_validation,
                                   pipeline=graph_based_pipe,
                                   parameters=params,
                                   scoring='accuracy')
