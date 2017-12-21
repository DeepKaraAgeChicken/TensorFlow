import os
import tarfile
import urllib.request

_url_base = 'http://www.cs.toronto.edu/%7Ekriz/'
_file_name = 'cifar-10-python.tar.gz'
_save_dir_name = 'data-set'

_expanded_dir_name = "cifar-10-batches-py"

_base_dir = os.path.dirname(os.path.abspath(__file__))
_data_dir = _base_dir + "/" + _save_dir_name
_saved_file = _base_dir + "/" + _save_dir_name + "/" + _file_name

dataset_dir = _data_dir + "/" + _expanded_dir_name

def download():
    """
    データセットをダウンロードします
    :return:
    """
    _download_file()
    _expand_data_file()

def _download_file():
    """
    URLからデータセットの圧縮ファイルをダウンロードします
    :return:
    """

    if os.path.exists(_saved_file):
        return

    os.mkdir(_base_dir + "/" + _save_dir_name)

    print("Downloading " + _file_name + " ... ")
    urllib.request.urlretrieve(_url_base + _file_name, _saved_file)
    print("Done")

    _expand_data_file()

    return 0

def _expand_data_file():
    """
    圧縮されたデータセットを解凍します
    :return:
    """
    if os.path.exists(_data_dir + "/" + _expanded_dir_name):
        return

    print("expanding " + _file_name + " ... ")
    tar = tarfile.open(_saved_file)
    tar.extractall(_data_dir)
    tar.close()
    print("Done")


if __name__ == '__main__':
    download()