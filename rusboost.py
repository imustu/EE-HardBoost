from collections import Counter
import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble._forest import BaseForest
from sklearn.preprocessing import normalize
from sklearn.tree import BaseDecisionTree
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



def plot_AUC(model,X_test,y_test):
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print(roc_auc)
class RandomUnderSampler(object):

    def __init__(self, with_replacement=True, return_indices=False,
                 random_state=None):
        self.return_indices = return_indices
        self.with_replacement = with_replacement
        self.random_state = random_state

    def sample(self, n_samples):
        """Perform undersampling.
        Parameters
        ----------
        n_samples : int
            Number of samples to remove.
        Returns
        -------
        S : array, shape = [n_majority_samples - n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        if self.n_majority_samples <= n_samples:
            n_samples = self.n_majority_samples

        idx = np.random.choice(self.n_majority_samples,
                               size=self.n_majority_samples - n_samples,
                               replace=self.with_replacement)

        if self.return_indices:
            return (self.X[idx], idx)
        else:
            return self.X[idx]

    def fit(self, X):
        """Train model based on input data.
        Parameters
        ----------
        X : array-like, shape = [n_majority_samples, n_features]
            Holds the majority samples.
        """
        self.X = X
        self.n_majority_samples, self.n_features = self.X.shape

        return self

class RUSBoost(AdaBoostClassifier):
    def __init__(self,
                 n_samples=100,
                 min_ratio=1.0,
                 with_replacement=True,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.min_ratio = min_ratio
        self.algorithm = algorithm
        self.rus = RandomUnderSampler(with_replacement=with_replacement,
                                      return_indices=True,
                                      random_state=random_state)

        super(RUSBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.

        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Random undersampling step.
            X_maj = X[np.where(y != self.minority_target)]
            X_min = X[np.where(y == self.minority_target)]
            self.rus.fit(X_maj)

            n_maj = X_maj.shape[0]
            n_min = X_min.shape[0]
            if n_maj - self.n_samples < int(n_min * self.min_ratio):
                self.n_samples = n_maj - int(n_min * self.min_ratio)
            X_rus, X_idx = self.rus.sample(self.n_samples)

            y_rus = y[np.where(y != self.minority_target)][X_idx]
            y_min = y[np.where(y == self.minority_target)]

            sample_weight_rus = \
                sample_weight[np.where(y != self.minority_target)][X_idx]
            sample_weight_min = \
                sample_weight[np.where(y == self.minority_target)]

            # Combine the minority and majority class samples.
            X = np.vstack((X_rus, X_min))
            y = np.append(y_rus, y_min)

            # Combine the weights.
            sample_weight = \
                np.append(sample_weight_rus, sample_weight_min).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))

            # X, y, sample_weight = shuffle(X, y, sample_weight,
            #                              random_state=random_state)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self

def rus_boost():
    clf1 = RUSBoost(n_estimators=20)
    clf1.fit(X_train, y_train)
    # clf1.fit(X_train.fillna(0), y_train)
    y_pred = clf1.predict(X_test)
    print('rus_boost_Acc: ', accuracy_score(y_test, y_pred))
    print('rus_boost_Precision: ', precision_score(y_test, y_pred, average='binary'))
    print('rus_boost_Recall: ', recall_score(y_test, y_pred, average='binary'))
    print('rus_boost_F1: ', f1_score(y_test, y_pred, average='binary'))
    print('rus_boost_AUC: ', roc_auc_score(y_test, y_pred))
    plot_AUC(clf1, X_test.fillna(0), y_test)
    # 混淆矩阵
    disp = confusion_matrix(clf1, X_test.fillna(0), y_test, display_labels=[0, 1], values_format='', cmap=plt.cm.Blues)
    disp.ax_.set_title('confusion matrix')
    plt.show()
    print(metrics.confusion_matrix(y_test, y_pred))
if __name__ == '__main__':
    pima = pd.read_excel('pima.xlsx')
    pima_label = pd.read_excel('pima_label.xlsx')
    X_train, X_test, y_train, y_test = train_test_split(pima, pima_label, test_size=0.2,
                                                        random_state=42)
    rus_boost()