import metric_learn
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import pandas as pd
import pickle
from loader import get_dataset
from sklearn.model_selection import train_test_split



_, _, y_train, y_test = get_dataset(
      data_path='data/dataset.pik',
)

X = np.load(open("x_pca_64.npy","rb"))

X_train, X_test = train_test_split(X, test_size=0.4, random_state=42)



for num_constraint in [200, 500, 1000, 2000, 5000, 10000, 20000]:
    metric_learner = metric_learn.LSML_Supervised(num_constraints=num_constraint)
    metric_learner.fit(X_train, y_train)

    new_X_test = metric_learner.transform(X_test)
    new_X_train = metric_learner.transform(X_train)

    print("metric trained", num_constraint)

    for n in [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]:
        clf = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
        clf.fit(new_X_train, y_train)
        # print("model created, begin predicting")
        score = metrics.accuracy_score(y_test, clf.predict(new_X_test))
        print(n, score)
