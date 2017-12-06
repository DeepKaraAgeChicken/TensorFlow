def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



if __name__ == '__main__':

    import os
    print(os.getcwd())

    data_group_1 = unpickle("cifar-10/cifar-10-batches-py/data_batch_1")

    labels = data_group_1[b'labels']
    from numpy import ndarray

    #: :type: ndarray
    data = data_group_1[b'data']

    print(data[0].size)
    print(data.shape[0])
