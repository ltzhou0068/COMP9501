import numpy as np
import matplotlib.pyplot as plt

markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X',
           '.', ',' '1', '2', '3', '4', '+', 'x', '|', '_']


# NCA
tmp_data = np.loadtxt("./LSML-PCA.csv", delimiter='\t')
labels = tmp_data[:, 0]
tmp_data = tmp_data[:, 1:] * 100
labels = [f"{int(label)}" for label in labels]
print(labels)
print(tmp_data)
x_len, y_len = tmp_data.shape

k_values = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
fig = plt.figure(figsize=(8, 6))
for i, y in enumerate(tmp_data):
    plt.plot(list(range(0, 11)), y, linewidth=2, marker=markers[i], markersize=5)
plt.legend(labels, fontsize=15, loc="lower right")
plt.xticks(range(0, 11), labels=k_values, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Acc(%)', fontsize=15)
plt.xlabel('K values', fontsize=15)
plt.savefig('LSML-PCA.png')
plt.show()



# # KNN
# tmp_data = np.loadtxt("./Data.csv", delimiter=',', skiprows=1)
# tmp_data = tmp_data[:, 1:]
# data = []
# x_len, y_len = tmp_data.shape
# # print(tmp_data)
# # print(x_len, y_len)
# # exit(0)
# for i in range(y_len):
#     data.append(tmp_data[:, i].tolist())
# # print(data)
# # exit(0)

# fig = plt.figure(figsize=(8, 6))
# x = list(range(x_len))
# for i, y in enumerate(data):
#     plt.plot(x, y, linewidth=2, marker=markers[i], markersize=5)
# plt.legend(["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average"], fontsize=15,)
# plt.xticks(range(0, 10), labels=range(1, 11), fontsize=12)
# plt.yticks(fontsize=12)
# plt.ylabel('Acc(%)', fontsize=15)
# plt.xlabel('K values', fontsize=15)
# plt.savefig("KNN_res.png")
# plt.show()
