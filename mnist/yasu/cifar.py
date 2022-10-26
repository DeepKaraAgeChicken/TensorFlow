import os

DATASETS_DIR=os.path.expanduser("~/.keras/datasets/cifar")

def load_data(dir=DATASETS_DIR):
    import glob
    path = download(dir)
    unzip(path, dir)
    datadir = os.path.join(dir, 'cifar-10-batches-py')
    batch_files = glob.glob1(datadir, 'data_batch_*')
    meta_file = glob.glob1(datadir, 'batches.meta')[0]

    batchdict1 = unpickle(os.path.join(datadir, batch_files[0]))
    label_names = unpickle(os.path.join(datadir, meta_file))
    print(label_names)
    print(batchdict1)

def download(dir):
    import urllib.request
    if not os.path.exists(dir):
        os.mkdir(dir)

    path = os.path.join(dir, "cifar-10-python.tar.gz")
    if not os.path.exists(path):
        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        urllib.request.urlretrieve(url, path)

    return path

def unzip(tar, outdir):
    import tarfile
    with tarfile.open(tar, 'r') as tf:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tf, path=DATASETS_DIR)

def unpickle(path):
    import pickle
    with open(path, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def main():
    load_data()

if __name__ == '__main__':
    main()
