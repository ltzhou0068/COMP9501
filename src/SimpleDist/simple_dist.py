from loader import get_dataset
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import datetime

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_dataset(
        data_path='data/dataset.pik',
    )
    print("Dataset Loaded, begin training")

    metric_trails = ['euclidean', 'manhattan', 'chebyshev', 'cosine']
    k_trials = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

    for met in metric_trails:
        for k in k_trials:
            start = datetime.datetime.now()
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=4, metric=met)
            knn.fit(X_train, y_train)

            acc = metrics.accuracy_score(y_test, knn.predict(X_test))
            print("k: {} acc: {}".format(k, acc))

            end = datetime.datetime.now()
            print("Run time: " + str(end - start))
