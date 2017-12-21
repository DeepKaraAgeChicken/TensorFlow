from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 64
num_classes = 10
epochs = 10

# the data, shuffled and split between train and test sets
# x_XXX=画像データ、y_XXX=教師データ
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# [0,255]の値を[0.0,1.0]に変換
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# 教師データをベクトルに変換
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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

# 学習
try:
    model.load_weights('keras/weight.hdf5')
except OSError:
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    model.save_weights('keras/weight.hdf5')

# 学習結果のテスト
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
