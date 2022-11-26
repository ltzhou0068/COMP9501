import sklearn
from metric_learn import MLKR
import sys
sys.path.append("../..")

from SupervisedMetricLearning.Common import X_train, X_test, y_train, y_test
from SupervisedMetricLearning.KNNmodel import knn_predict
import numpy as np

# [0.8762140799785653, 0.8763480474244758, 0.8805010382477058, 
# 0.8774867707147164, 0.8800321521870186, 0.8794292986804206, 
# 0.87929533123451, 0.8807019894165717, 0.8761470962556099, 
# 0.8705874472503182]

def mlkr_method():
    k_values = list(range(3, 11)) + [15, 20]
    print(X_train.shape, y_train.shape)
    transformer = MLKR(n_components=50, verbose=True, tol=0.1)
    transformer.fit(X_train, y_train)
    X_train_comp = transformer.transform(X_train)
    X_test_comp = transformer.transform(X_test)
    print("MLKR finished")
    acc_list = []
    for k in k_values:
        acc = knn_predict(X_train_comp, X_test_comp, y_train, y_test, k)
        acc_list.append(acc)
    print(acc_list)


if __name__ == '__main__':
    mlkr_method()