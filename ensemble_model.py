import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from numpy import inner
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import  recall_score, f1_score, precision_score, accuracy_score
from imblearn.metrics import geometric_mean_score
import sklearn.metrics as metrics
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

class SP_AdaBoostClassifier(AdaBoostClassifier):

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict_proba = estimator.predict_proba(X)

        # print(y_predict_proba)
        # print(y_predict_proba.shape)

        pt = y_predict_proba[:, 1]
        pt = pt.astype('float')
        # print(pt)
        # print(pt.shape)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)
        # print(y)
        # print(y_predict)

        # f_num = pd.Series(y_predict)
        # print(f_num.value_counts())

        incorrect = y_predict != y
        # print(incorrect)

        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        proba = y_predict_proba  # alias for readability
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps

        estimator_weight = (-1. * self.learning_rate
                            * (((n_classes - 1.) / n_classes) *
                               inner(y_coding, np.log(y_predict_proba))))

        # 样本更新的公式，只需要改写这里
        if not iboost == self.n_estimators - 1:
            sample_weight *= (

                  #ee_boost_result(y, y_predict, estimator_error, self.n_estimators) *
                                     np.exp(estimator_weight *
                                     ((sample_weight > 0) |
                                      (estimator_weight < 0)) * self._beta(y, y_predict, pt)

                                     ))
        # print(sample_weight)# 在原来的基础上乘以self._beta(y, y_predict)，即代价调整函数
        return sample_weight, 1., estimator_error

    #  新定义的代价调整函数
    def _beta(self, y, y_hat, pt_):
        res = []
        for i in zip(y, y_hat, pt_):
            if i[0] == i[1] and i[1] == 1:
                res.append(i[2] ** 2)
            elif i[0] == 1 and i[1] == 0:
                res.append((1 - i[2]) ** 2)
            elif i[0] == 0 and i[1] == 0:
                res.append((1 - i[2]) ** 2)
            else:
                res.append(i[2] ** 2)
                # print(i[0], i[1])
        return np.array(res, dtype=np.float64)

def adaboost():
    clf1 = AdaBoostClassifier(learning_rate=1.0,n_estimators=10)
    clf1.fit(X_train, y_train)
    # clf1.fit(X_train.fillna(0), y_train)
    y_pred = clf1.predict(X_test)
    probs = clf1.predict_proba(X_test)
    preds = probs[:, 1]
    print('adaboost_Acc: ', accuracy_score(y_test, y_pred))
    print('adaboost_Precision: ', precision_score(y_test, y_pred, average='binary'))
    print('adaboost_Recall: ', recall_score(y_test, y_pred, average='binary'))
    print('adaboost_F1: ', f1_score(y_test, y_pred, average='binary'))
    print('adaboost_Gmean: ' , geometric_mean_score(y_test, y_pred))
    print('adaboost_AUC: ', roc_auc_score(y_test, preds))
    # plot_AUC(clf1, X_test.fillna(0), y_test)
    # # 混淆矩阵
    # disp = plot_confusion_matrix(clf1, X_test.fillna(0), y_test,
    #                              display_labels=[0, 1],
    #                              values_format='',
    #                              cmap=plt.cm.Blues
    #                              )
    # disp.ax_.set_title('confusion matrix')
    # plt.show()
    print(metrics.confusion_matrix(y_test, y_pred))
def SP_AdaBoost():
    clf2 = SP_AdaBoostClassifier(learning_rate=1.0 , n_estimators=10)
    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    probs = clf2.predict_proba(X_test)
    preds = probs[:, 1]
    print('0urmethord_Acc: ', accuracy_score(y_test, y_pred))
    print('0urmethord_Precision: ', precision_score(y_test, y_pred, average='binary'))
    print('0urmethord_Recall: ', recall_score(y_test, y_pred, average='binary'))
    print('0urmethord_F1: ', f1_score(y_test, y_pred, average='binary'))
    print('0urmethord_Gmean: ' , geometric_mean_score(y_test, y_pred))
    print('0urmethord_AUC: ', roc_auc_score(y_test, preds ))
    # plot_AUC(clf2, X_test.fillna(0), y_test)
    # # 混淆矩阵
    # disp = plot_confusion_matrix(clf2, X_test.fillna(0), y_test,
    #                              display_labels=[0, 1],
    #                              values_format='',
    #                              cmap=plt.cm.Blues
    #                              )
    # disp.ax_.set_title('confusion matrix')
    # plt.show()
    print(metrics.confusion_matrix(y_test, y_pred))

def easy_ensemble_ada():
    clf2 = EasyEnsembleClassifier(n_estimators=120, random_state=30, replacement=True ,
                                  base_estimator=AdaBoostClassifier(
                                                                    learning_rate=1.0,
                                                                     n_estimators=20, random_state=0))

    clf2.fit(X_train, y_train)

    y_pred = clf2.predict(X_test)
    probs = clf2.predict_proba(X_test)
    preds = probs[:, 1]

    #print(y_pred)
    print('EE_ada_Acc: ', accuracy_score(y_test, y_pred))
    print('EE_ada_Precision: ', precision_score(y_test, y_pred, average='binary'))
    print('EE_ada_Recall: ', recall_score(y_test, y_pred, average='binary'))
    print('EE_ada_F1: ', f1_score(y_test, y_pred, average='binary'))
    print('EE_ada_Gmean: ' , geometric_mean_score(y_test, y_pred))
    print('EE_ada_AUC: ', roc_auc_score(y_test, preds))
    # plot_AUC(clf2, X_test.fillna(0), y_test)
    # # 混淆矩阵
    # disp = plot_confusion_matrix(clf2, X_test.fillna(0), y_test,
    #                              display_labels=[0, 1],
    #                              values_format='',
    #                              cmap=plt.cm.Blues
    #                              )
    # disp.ax_.set_title('confusion matrix')
    # plt.show()
    print(metrics.confusion_matrix(y_test, y_pred))
def easy_ensemble_SP_Ada():
    clf2 = EasyEnsembleClassifier(n_estimators=120, random_state=10, replacement=True ,
                                   #                           base_estimator=XGBClassifier(objective="binary:logistic",random_state=42))
                                   #                               base_estimator = GradientBoostingClassifier(max_depth=5,n_estimators=100 , loss="deviance"))

                                  base_estimator=SP_AdaBoostClassifier(
                                                                    learning_rate=1.0,
                                                                     n_estimators=20, random_state=0))

    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    probs = clf2.predict_proba(X_test)
    preds = probs[:, 1]
    print('EE_coat_Acc: ', accuracy_score(y_test, y_pred))
    print('EE_coat_Precision: ', precision_score(y_test, y_pred, average='binary'))
    print('EE_coat_Recall: ', recall_score(y_test, y_pred, average='binary'))
    print('EE_coat_F1: ', f1_score(y_test, y_pred, average='binary'))
    print('EE_coat_Gmean: ' , geometric_mean_score(y_test, y_pred))
    print('EE_coat_AUC: ', roc_auc_score(y_test, preds))
    # plot_AUC(clf2, X_test.fillna(0), y_test)
    # # 混淆矩阵
    # disp = plot_confusion_matrix(clf2, X_test.fillna(0), y_test,
    #                              display_labels=[0, 1],
    #                              values_format='',
    #                              cmap=plt.cm.Blues
    #                              )
    # disp.ax_.set_title('confusion matrix')
    # plt.show()
    print(metrics.confusion_matrix(y_test, y_pred))

def bagging_RF():
    clf1 = BaggingClassifier(RandomForestClassifier(),
                                n_estimators=500, bootstrap=True, n_jobs=-1, oob_score=True)
    clf1.fit(X_train, y_train)
    # clf1.fit(X_train.fillna(0), y_train)
    y_pred = clf1.predict(X_test)
    probs = clf1.predict_proba(X_test)
    preds = probs[:, 1]
    print('rus_boost_Acc: ', accuracy_score(y_test, y_pred))
    print('rus_boost_Precision: ', precision_score(y_test, y_pred, average='binary'))
    print('rus_boost_Recall: ', recall_score(y_test, y_pred, average='binary'))
    print('rus_boost_F1: ', f1_score(y_test, y_pred, average='binary'))
    print('rus_boost_Gmean: ' , geometric_mean_score(y_test, y_pred))
    print('rus_boost_AUC: ', roc_auc_score(y_test, preds))
    # plot_AUC(clf1, X_test.fillna(0), y_test)
    # # 混淆矩阵
    # disp = plot_confusion_matrix(clf1, X_test.fillna(0), y_test,
    #                              display_labels=[0, 1],
    #                              values_format='',
    #                              cmap=plt.cm.Blues
    #                              )
    # disp.ax_.set_title('confusion matrix')
    # plt.show()
    print(metrics.confusion_matrix(y_test, y_pred))
def select_n_estimatorsAUC():
    # 确定n_estimators的取值范围
    tuned_parameters = range(10, 160, 10)
    # 创建添加accuracy的一个numpy
    roc_auc = np.zeros(len(tuned_parameters))
    roc_auc_list1 = []
    roc_auc_list2 = []
    f1_list1 = []
    f1_list2 = []
    g_mean_list1 = []
    g_mean_list2 = []
    for j, one_parameter in enumerate(tuned_parameters):
        # model1 = EasyEnsembleClassifier(n_estimators=one_parameter, random_state=0, replacement=True,
        # #                               base_estimator=xgb.XGBClassifier(objective="binary:logistic", random_state=42))
        # #                                base_estimator = GradientBoostingClassifier(max_depth=5,n_estimators=100 , loss="deviance"))
        #
        # base_estimator=AdaBoostClassifier(n_estimators=20, random_state=0 , learning_rate=1.0))
        # model1.fit(X_train.fillna(0), y_train)
        # probs = model1.predict_proba(X_test)
        # y_pred = model1.predict(X_test)
        # preds = probs[:, 1]
        # fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        # roc_auc = metrics.auc(fpr, tpr)
        # f1 = metrics.f1_score(y_test, y_pred, average='binary')
        # g_mean = geometric_mean_score(y_test, y_pred, average='binary')
        #
        # roc_auc_list1.append(roc_auc)
        # f1_list1.append(f1)
        # g_mean_list1.append(g_mean)

        model2 = EasyEnsembleClassifier(n_estimators=one_parameter, random_state=0, replacement=True,
        #                               base_estimator=xgb.XGBClassifier(objective="binary:logistic", random_state=42))
        #                                base_estimator = GradientBoostingClassifier(max_depth=5,n_estimators=100 , loss="deviance"))

        base_estimator=AdaBoostClassifier(n_estimators=20, random_state=20, learning_rate=1.0))
        model2.fit(X_train.fillna(0), y_train)
        probs = model2.predict_proba(X_test)
        y_pred = model2.predict(X_test)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)
        f1 = metrics.f1_score(y_test, y_pred, average='binary')
        g_mean = geometric_mean_score(y_test, y_pred, average='binary')

        roc_auc_list2.append(roc_auc)
        f1_list2.append(f1)
        g_mean_list2.append(g_mean)

    print(f1_list1)
    print(g_mean_list1)
    print(roc_auc_list1)

    print(f1_list2)
    print(g_mean_list2)
    print(roc_auc_list2)

# 优化结果过程可视化
    #fig, axes = plt.subplots(figsize=(7, 7), dpi=200)

    plt.plot(tuned_parameters, roc_auc_list1, label='AdaBoost', linewidth = 3.0 ,
             color='red', linestyle='-', marker='.', markersize=8)
    plt.plot(tuned_parameters, roc_auc_list2, label='HardBoost', linewidth = 3.0 ,
             color='black', linestyle='--', marker='+', markersize=8)

    #plt.plot(tuned_parameters, roc_auc_list)
    plt.xlabel('The number of base classifiers')
    plt.ylabel('AUC')
    #plt.title('Boosting Rounds = 50')
    plt.legend(loc = 'upper left')
    plt.show()
def select_n_estimatorsF1():
    tuned_parameters = range(10, 150, 10)
    roc_auc = np.zeros(len(tuned_parameters))
    roc_auc_list1 = []
    roc_auc_list2 = []
    for j, one_parameter in enumerate(tuned_parameters):
        model1 = EasyEnsembleClassifier(n_estimators=one_parameter, random_state=0, replacement=True,
        base_estimator=AdaBoostClassifier(n_estimators=30, random_state=0 , learning_rate=1.0))
        model1.fit(X_train.fillna(0), y_train)
        probs = model1.predict(X_test)
        #preds = probs[:, 1]
        roc_auc = metrics.f1_score(y_test, probs, average='binary')
        #roc_auc = metrics.f1_score(fpr, tpr)
        roc_auc_list1.append(roc_auc)

        # model2 = EasyEnsembleClassifier(n_estimators=one_parameter, random_state=0, replacement=True,
        # base_estimator=AdaCostClassifier(n_estimators=5, random_state=0, learning_rate=1.0))
        # model2.fit(X_train.fillna(0), y_train)
        # probs = model2.predict(X_test)
        # #preds = probs[:, 1]
        # roc_auc= metrics.f1_score(y_test, probs, average='binary')
        # #roc_auc = metrics.f1_score(fpr, tpr)
        # roc_auc_list2.append(roc_auc)
    print(roc_auc_list1)
    print(roc_auc_list2)
# 优化结果过程可视化
    plt.plot(tuned_parameters, roc_auc_list1, label='AdaBoost', linewidth = 1.6 ,
             color='red', linestyle='-', marker='.', markersize=8)
    plt.plot(tuned_parameters, roc_auc_list2, label='HardBoost', linewidth = 1.6 ,
             color='black', linestyle='--', marker='+', markersize=8)
    plt.xlabel('The number of base classifiers')
    plt.ylabel('AUC')
    #plt.title('Boosting Rounds = 50')
    plt.legend(loc = 'upper left')
    plt.show()
def select_n_estimatorsGmean():
    tuned_parameters = range(10, 150, 10)
    roc_auc = np.zeros(len(tuned_parameters))
    roc_auc_list1 = []
    roc_auc_list2 = []
    for j, one_parameter in enumerate(tuned_parameters):
        model1 = EasyEnsembleClassifier(n_estimators=one_parameter, random_state=0, replacement=True,
        base_estimator=AdaBoostClassifier(n_estimators=10, random_state=0 , learning_rate=1.0))
        model1.fit(X_train.fillna(0), y_train)
        probs = model1.predict(X_test)
        # preds = probs[:, 1]
        roc_auc = geometric_mean_score(y_test, probs, average='binary')
        # roc_auc = metrics.f1_score(fpr, tpr)
        roc_auc_list1.append(roc_auc)

        # model2 = EasyEnsembleClassifier(n_estimators=one_parameter, random_state=0, replacement=True,
        # base_estimator=AdaCostClassifier(n_estimators=10, random_state=0, learning_rate=1.0))
        # model2.fit(X_train.fillna(0), y_train)
        # probs = model2.predict(X_test)
        # # preds = probs[:, 1]
        # roc_auc = geometric_mean_score(y_test, probs, average='binary')
        # # roc_auc = metrics.f1_score(fpr, tpr)
        # roc_auc_list2.append(roc_auc)
    print(roc_auc_list1)
    print(roc_auc_list2)
# 优化结果过程可视化
    plt.plot(tuned_parameters, roc_auc_list1, label='AdaBoost', linewidth = 1.6 ,
             color='red', linestyle='-', marker='.', markersize=8)
    plt.plot(tuned_parameters, roc_auc_list2, label='HardBoost', linewidth = 1.6 ,
             color='black', linestyle='--', marker='+', markersize=8)
    plt.xlabel('The number of base classifiers')
    plt.ylabel('AUC')
    #plt.title('Boosting Rounds = 50')
    plt.legend(loc = 'upper left')
    plt.show()




if __name__ == '__main__':
    X = pd.read_excel('data/features_all.xlsx')
    y = pd.read_excel('data/label_all.xlsx')
    sm = BorderlineSMOTE(random_state=42, k_neighbors=5, kind="borderline-1", sampling_strategy={1: 500})
    X, y = sm.fit_resample(X, y)
    print(y['label'].value_counts())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                         random_state=7)
    print("----------")
    #choose one method
    #adaboost()
    # SP_AdaBoost()
    easy_ensemble_ada()
    # easy_ensemble_SP_Ada()
    #select_n_estimatorsAUC()
    #select_n_estimatorsF1()
    #select_n_estimatorsGmean()