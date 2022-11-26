from sklearn.svm import SVC
from loader import get_dataset
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_dataset(
        data_path='data/dataset.pik',
    )
    print("Dataset Loaded, begin training")

    # knn = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    # knn.fit(X_train, y_train)
    #
    # print("Training Finished")
    #
    # acc = metrics.accuracy_score(y_test, knn.predict(X_test))
    # print(acc)

    cv_scores = []
    for k in range(1, 30):
        print("K=" + str(k))
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=4)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        print(scores)
        cv_scores.append(scores.mean())

    plt.plot(range(1, 30), cv_scores)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()
