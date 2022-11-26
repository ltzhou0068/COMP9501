import numpy as np
import matplotlib.pyplot as plt

markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
           '.', ',' '1', '2', '3', '4', '+', 'x', '|', '_']


def simple():
    tmp_data = np.loadtxt("./sim_dist.csv", delimiter=',')
    labels = ['Manhattan', 'Euclidean', 'Chebyshev', 'Cosine']
    print(labels)
    print(tmp_data.shape)
    x_len, y_len = tmp_data.shape

    k_values = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    fig = plt.figure(figsize=(8, 6))
    for i, y in enumerate(tmp_data):
        plt.plot(list(range(0, 10)), y, linewidth=2, marker=markers[i], markersize=5)
    plt.legend(labels, fontsize=15)
    plt.xticks(range(0, 10), labels=k_values, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Acc(%)', fontsize=15)
    plt.xlabel('K values', fontsize=15)
    plt.savefig('simple.png')
    plt.show()


def cv():
    tmp_data = np.loadtxt("./cv.csv", delimiter=',')
    x_len, y_len = tmp_data.shape
    print(x_len, y_len)
    data = []

    for i in range(x_len):
        data.append(np.mean(tmp_data[i])*100)

    fig = plt.figure(figsize=(8, 6))
    x = list(range(x_len))

    plt.plot(x, data, linewidth=2, marker=markers[0], markersize=5)
    # plt.legend(["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average"], fontsize=15, )
    # plt.xticks(range(0, 29), labels=range(1, 30), fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Acc(%)', fontsize=15)
    plt.xlabel('K values', fontsize=15)
    plt.savefig("cv1.png")
    plt.show()


cv()
