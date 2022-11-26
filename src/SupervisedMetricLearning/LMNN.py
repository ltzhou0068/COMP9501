import sklearn
# from metric_learn import LMNN
from sklearn.decomposition import KernelPCA
import sys
sys.path.append("../..")

from SupervisedMetricLearning.Common import X_train, y_train, X_test, y_test
from SupervisedMetricLearning.KNNmodel import knn_predict
from pylmnn import LargeMarginNearestNeighbor as LMNN
import numpy as np

# [0.8758791613637886, 0.875946145086744, 0.8795632661263313,
#  0.8765489985933418, 0.8806350056936164, 0.8782905753901802,
#  0.8800991359099739, 0.8790273963426887, 0.8748074217965035, 0.869917610020765]


def lmnn_method():
    k_values = list(range(3, 11)) + [15, 20]
    print(X_train.shape, y_train.shape)
    acc_list = []
    # pca_transformer = KernelPCA(n_components=20, n_jobs=4, kernel='cosine')
    # X_train_decomp = pca_transformer.fit_transform(X_train)
    # X_test_decomp = pca_transformer.transform(X_test)
    # print("PCA Finished")
    transformer = LMNN(tol=0.1, verbose=True)
    print("LMNN Start")
    transformer.fit(X_train, y_train)
    X_train_comp = transformer.transform(X_train)
    X_test_comp = transformer.transform(X_test)
    print(X_train_comp.shape, X_test_comp.shape)
    for k in k_values:
        acc = knn_predict(X_train, X_test, y_train, y_test, k)
        acc_list.append(acc)
    print(acc_list)


if __name__ == '__main__':
    lmnn_method()