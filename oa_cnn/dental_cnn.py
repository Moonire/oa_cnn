import keras
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, LeakyReLU
from keras.optimizers import Adam

from PIL import Image
import numpy as np


def imread(img):
    size = img.size[0]
    M = np.zeros((size, size, 3))

    for i in range(size):
        for j in range(size):
            for k in range(3):
                M[i][j][k] = img.getpixel((i, j))[k]/255
    return M


def resize_image(file):
    img = Image.open(file)
    img = img.resize((52, 52), Image.ANTIALIAS)

    return img


def matrix_to_picture(data, size):
    img = Image.new('RGB', (size, size))

    for m, i in enumerate(data):
        for n, j in enumerate(i):
            img.putpixel((m, n), tuple((j*255).astype(int)))

    img.show()


def dcnn(M, n, deapth=32):
    M.add(Conv2D(filters=deapth, kernel_size=(5, 5), padding='same', input_shape=(n, n, 3)))
    M.add(LeakyReLU())

    M.add(Conv2D(filters=deapth//2, kernel_size=(4, 4), padding='same'))
    M.add(LeakyReLU())

    M.add(Conv2D(filters=deapth//4, kernel_size=(3, 3), padding='same'))
    M.add(LeakyReLU())

    M.add(Conv2D(filters=deapth//8, kernel_size=(3, 3), padding='same'))
    M.add(LeakyReLU())

    M.add(Conv2D(filters=3, kernel_size=(3, 3), padding='same'))
    M.add(LeakyReLU())


def train(size, e, g):

    data = np.expand_dims(imread(resize_image(e)), axis=0)
    labels = np.expand_dims(imread(resize_image(g)), axis=0)

    x_train, x_test = data, data
    y_train, y_test = labels, labels

    model = Sequential()
    dcnn(model, size)

    checkpoint = ModelCheckpoint('model.h5', verbose=0, monitor='loss', save_best_only=True, mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-02, decay=1e-4))
    model.fit(x_train, y_train, epochs=1_000, batch_size=20, callbacks=[checkpoint])
    score = model.evaluate(x_test, y_test, batch_size=20)
    print(score)


def load(x_train, y_train, id_=0, size=20, show=True):
    model = load_model('model_000434.h5')
    prediction = model.predict(x_train[id_].reshape((1, size, size, 3)))[0]

    if show:
        matrix_to_picture(y_train[id_], 52)
        matrix_to_picture(prediction, 52)
