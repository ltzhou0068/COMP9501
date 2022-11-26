import sklearn
from metric_learn import NCA
import sys
sys.path.append("../..")

from zSupervisedMetricLearning.Common import X_train, X_test, y_train, y_test
from SupervisedMetricLearning.KNNmodel import knn_predict
import numpy as np

# NCA = 
# [0.8794292986804206, 0.8805680219706611, 0.8843191104561592,
#  0.8822426150445442, 0.8851898988545783, 0.8838502243954719,
#  0.8854578337463996, 0.8843191104561592, 0.8811038917543037,
#  0.8767499497622078]

# [0.880969924308393, 0.8827115011052314, 0.889074954785987, 
# 0.8884721012793891, 0.8919552548730658, 0.8914193850894233, 
# 0.8939647665617255, 0.8926920758255744, 0.8922901734878425, 
# 0.8896108245696296]


def nca_method():
    k_values = list(range(3, 11)) + [15, 20]
    print(X_train.shape, y_train.shape)
    transformer = NCA(n_components=50, verbose=True, tol=0.1)
    transformer.fit(X_train, y_train)
    X_train_comp = transformer.transform(X_train)
    X_test_comp = transformer.transform(X_test)
    acc_list = []
    for k in k_values:
        acc = knn_predict(X_train_comp, X_test_comp, y_train, y_test, k)
        acc_list.append(acc)
    print(acc_list)


if __name__ == '__main__':
    nca_method()