import matplotlib.pylab as plt
import pandas as pd
from keras.callbacks import Callback
from seaborn import heatmap
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

plt.rcParams['figure.figsize'] = (16, 8)


def create_clf_report(y_true, y_pred, classes):
    """
    This function calculates several metrics about a classifier and creates a mini report.
    :param y_true: iterable. An iterable of string or ints.
    :param y_pred: iterable. An iterable of string or ints.
    :param classes: iterable. An iterable of string or ints.
    :return: dataframe. A pandas dataframe with the confusion matrix.
    """
    confusion = pd.DataFrame(confusion_matrix(y_true, y_pred),
                             index=classes,
                             columns=['predicted_{}'.format(c) for c in classes])

    print("-" * 80, end='\n')
    print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_true, y_pred) * 100))
    print("-" * 80)

    print("Confusion Matrix:", end='\n\n')
    print(confusion)

    print("-" * 80, end='\n')
    print("Classification Report:", end='\n\n')
    print(classification_report(y_true, y_pred, digits=3), end='\n')

    return confusion


def print_confusion_matrix(y_true, y_pred, labels, size=(10, 10)):
    """

    :param y_true:
    :param y_pred:
    :param labels:
    :param size:
    :return:
    """

    conf_mat = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=size)

    h = heatmap(conf_mat,
                annot=True,
                fmt='d')

    h.set_xticklabels(labels, rotation=90)
    h.set_yticklabels(labels, rotation=0)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return pd.DataFrame(conf_mat,
                        columns=["pred_{}".format(l) for l in labels],
                        index=["true_{}".format(l) for l in labels])


class KerasRocCallback(Callback):
    """
    """

    def __init__(self, X_train, X_val, y_train, y_val):
        """

        :param X_train:
        :param X_val:
        :param y_train:
        :param y_val:
        """
        super().__init__()

        self.x = X_train
        self.y = y_train
        self.x_val = X_val
        self.y_val = y_val

    def on_train_begin(self, logs=None):
        """

        :param logs:
        :return:
        """
        if logs is None:
            logs = {}
        return

    def on_train_end(self, logs=None):
        """

        :param logs:
        :return:
        """
        if logs is None:
            logs = {}
        return

    def on_epoch_begin(self, epoch, logs=None):
        """

        :param epoch:
        :param logs:
        :return:
        """
        if logs is None:
            logs = {}
        return

    def on_epoch_end(self, epoch, logs=None):
        """

        :param epoch:
        :param logs:
        :return:
        """
        if logs is None:
            logs = {}

        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: {} - roc-auc_val: {}'.format(str(round(roc, 4)),
                                                       str(round(roc_val, 4))),
              end=100 * ' ' + '\n')

        return

    def on_batch_begin(self, batch, logs=None):
        """

        :param batch:
        :param logs:
        :return:
        """
        if logs is None:
            logs = {}
        return

    def on_batch_end(self, batch, logs=None):
        """

        :param batch:
        :param logs:
        :return:
        """
        if logs is None:
            logs = {}
        return


def plot_roc_curve(y_true, y_pred_scores, pos_label=1):
    """

    :param y_true:
    :param y_pred_scores:
    :param pos_label:
    :return:
    """

    fpr, tpr, _ = roc_curve(y_true, y_pred_scores, pos_label=pos_label)
    roc_auc = roc_auc_score(y_true, y_pred_scores)
    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_scores, pos_label=1):
    """

    :param y_true:
    :param y_pred_scores:
    :param pos_label:
    :return:
    """

    average_precision = average_precision_score(y_true, y_pred_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_scores, pos_label=pos_label)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()
