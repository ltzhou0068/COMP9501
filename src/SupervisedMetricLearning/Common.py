import sys
sys.path.append("../..")

from loader import get_dataset
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = get_dataset(
    data_path='data/dataset.pik',
)

standard = StandardScaler()
X_train = standard.fit_transform(X_train)
X_test = standard.fit_transform(X_test)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)