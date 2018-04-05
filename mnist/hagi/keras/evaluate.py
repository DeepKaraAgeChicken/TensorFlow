from typing import Callable

import keras
import numpy as np
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.preprocessing import image

batch_size = 64
epochs = 10
num_classes = 10


def main():
    predict = generate_predict('weight.hdf5')
    file_name = '../MNIST_data/images/image_1_11152.jpg'
    # predict
    result = predict(file_name)
    print(result)
    # picture
    img = image.load_img(file_name, target_size=(28, 28), grayscale=True)
    # img->numpy.array
    x = image.img_to_array(img)
    x = x.reshape((784,))
    x /= 255.0
    for i in range(28):
        for j in range(28):
            if x[i * 28 + j] > 0.5:
                print("+", end="")
            else:
                print(" ", end="")
        print()


def generate_predict(file: str) -> Callable[[str], np.ndarray]:
    model = get_model()
    model.load_weights(file)

    def closure(file_name):
        img = image.load_img(file_name, target_size=(28, 28), grayscale=True)
        x = image.img_to_array(img)
        x = x.reshape((784,))
        x /= 255.0
        preds = model.predict(np.array([x]), 1)
        return preds[0]

    return closure


def calc_weight(model):
    # x_XXX=画像データ、y_XXX=教師データ
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # [0,255]の値を[0.0,1.0]に変換
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))


def get_model():
    """
    :return:
    """
    model = Sequential()
    # 隠れ層は2層
    # Dense: 全結合のニューラルネットワークレイヤ
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    # model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.2))
    # 回帰問題なのでソフトマックス
    model.add(Dense(num_classes, activation='softmax'))

    # 隠れ層のサマリ表示
    model.summary()

    # 出力層
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model


def fit_model(model, x_train, y_train, x_test, y_test):
    return model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(x_test, y_test))


if __name__ == '__main__':
    main()
