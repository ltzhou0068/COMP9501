from sklearn.neighbors import KNeighborsClassifier

def knn_predict(X_train, X_test, y_train, y_test, k):
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=4)
    model.fit(X_train, y_train)
    # print("train finished")
    acc = model.score(X_test, y_test)
    print("k = %d, score = %f" % (k, acc))
    return acc