from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from rpcc.evaluation import create_clf_report
from rpcc.load_data import DataLoader, restore_data_loader


def custom_split(text):
    """

    :param text:
    :return:
    """
    if text is None:
        return []
    else:
        return text.split(',')


if __name__ == "__main__":
    obj = restore_data_loader()

    authors_train = obj.x_train['authors'].fillna('')
    authors_val = obj.x_val['authors'].fillna('')

    y_train = obj.y_train
    y_val = obj.y_val

    vec = CountVectorizer(tokenizer=DataLoader.clean_up_authors, min_df=0, ngram_range=(1, 2))
    X_train = vec.fit_transform(authors_train)
    X_val = vec.transform(authors_val)

    print("\nTrain matrix dimensionality: ", X_train.shape)
    print("Validation matrix dimensionality: ", X_val.shape)

    # Use logistic regression to classify the articles of the test set
    clf = MultinomialNB(alpha=0.45)  # scores acc 30.60
    # clf = LogisticRegression(class_weight=None, C=1, multi_class='ovr')  # scores acc 29.13
    # clf = SVC(kernel='linear', C=0.5, probability=True) # 29.03
    clf = RandomForestClassifier(n_estimators=50, criterion='entropy')  # 29.03

    clf.fit(X_train, y_train)

    y_val_pred_probs = clf.predict_proba(X_val)
    y_val_pred_labels = clf.predict(X_val)

    report = create_clf_report(y_val, y_val_pred_labels, clf.classes_)

    print(report)
