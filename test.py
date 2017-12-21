import tensorflow as tf
import numpy as np

"""
li = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
print(li)
nl = np.vstack(li)
print(nl)

rli = [1, 2, 3, 4, 5]
nr = np.array(rli)[:, np.newaxis]
print(np.newaxis)
print(nr)
"""


def array_data_sample(data, batch_size):  # data: list of numpy array
    import random
    data_lens = len(data[0])
    index = random.sample(range(data_lens), batch_size)

    batch = []
    for i in range(len(data)):
        sample = []
        for j in range(batch_size):
            sample.append(data[i][index[j]])
        batch.append(np.vstack(sample))

    return batch


def test():
    sli = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]
    nsli = np.vstack(sli)  # np.array(sli)[:, np.newaxis]
    # print(nsli)
    ali = [[1, 2], [4, 5], [7, 8], [10, 11], [13, 14], [16, 17]]
    nali = np.vstack(ali)
    rli = [1, 2, 3, 4, 5, 6]
    # nrli = np.array(rli)[:, np.newaxis]
    nrli = np.vstack(rli)
    # print(nrli)

    data = [nsli, nali, nrli]

    batch = array_data_sample(data, 3)
    # print(batch)


a = [[[1, 2], [2, 3], [3, 4]], [[2, 3], [3, 4], [4, 5]]]
b = a[0]
