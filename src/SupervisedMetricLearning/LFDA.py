import sklearn
from metric_learn import LFDA
import sys
sys.path.append("../..")

from SupervisedMetricLearning.Common import X_train, X_test, y_train, y_test
from SupervisedMetricLearning.KNNmodel import knn_predict
import numpy as np

# [0.8632862214481881, 0.861812579543171, 0.8667693750418648,
#  0.8633532051711434, 0.8650277982450265, 0.8632192377252328,
#  0.8630182865563668, 0.8620805144349923, 0.8592002143479135,
#  0.8551811909705942]

# [0.9095049902873602, 0.9092370553955389, 0.9148636881237859,
#  0.9147297206778753, 0.9169401835354009, 0.9164043137517583,
#  0.9169401835354009, 0.9179449393797308, 0.9188827115011052,
#  0.9186147766092839]


def lfda_method():
    k_values = list(range(3, 11)) + [15, 20]
    print(X_train.shape, y_train.shape)
    transformer = LFDA(n_components=50)
    transformer.fit(X_train, y_train)
    X_train_comp = transformer.transform(X_train)
    X_test_comp = transformer.transform(X_test)
    print("LFDA finished")
    acc_list = []
    for k in k_values:
        acc = knn_predict(X_train_comp, X_test_comp, y_train, y_test, k)
        acc_list.append(acc)
    print(acc_list)


if __name__ == '__main__':
    lfda_method()