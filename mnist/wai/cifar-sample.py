from mnist.wai import cifar10


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



if __name__ == '__main__':
    import os
    print(os.getcwd())

    cifar10.download()
    data_group_1 = unpickle(cifar10.dataset_dir + "/data_batch_1")

    labels = data_group_1[b'labels']
    from numpy import ndarray

    #: :type: ndarray
    data = data_group_1[b'data']

    print(data[0].size)
    print(data.shape[0])
