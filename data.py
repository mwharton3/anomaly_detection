import os
from scipy import misc
import cv2
import numpy as np
from tqdm import tqdm_notebook as tqdm
import keras.backend as K
import random
import warnings
import multiprocessing
import ctypes as c
import random
warnings.filterwarnings("ignore")


def generateTestImages():
    # Set up
    posdir = '/home/mwharton/Data/potholesSimplex/test/positive'
    negdir = '/home/mwharton/Data/potholesSimplex/test/negative'
    npos = len(os.listdir(posdir))
    nneg = len(os.listdir(negdir))
    x_test_pos = np.zeros((npos, 512, 512, 3))
    x_test_neg = np.zeros((nneg, 512, 512, 3))
    y_test_pos = np.zeros((npos, 2))
    y_test_neg = np.zeros((nneg, 2))
    y_test_pos[:, 1] = 1
    y_test_neg[:, 0] = 1

    # Read the images
    i = 0
    for fn in tqdm(os.listdir(posdir)):
        fullpath = os.path.join(posdir, fn)
        im = np.flip(cv2.imread(fullpath), axis=2)
        x_test_pos[i, :, :, :] = misc.imresize(im, (512, 512)) / 255.0
        i += 1
    i = 0
    for fn in tqdm(os.listdir(negdir)):
        fullpath = os.path.join(negdir, fn)
        im = np.flip(cv2.imread(fullpath), axis=2)
        x_test_neg[i, :, :, :] = misc.imresize(im, (512, 512)) / 255.0
        i += 1

    x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
    y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

    return x_test, y_test



# Load a single random image with a random classification
def generateImage(x_train, y_train, ind):
    posdir = '/home/mwharton/Data/potholesSimplex/train/positive'
    negdir = '/home/mwharton/Data/potholesSimplex/train/negative'
    ind = int(ind)
    if random.random() < 0.7:  # Number of files dictates approximate odds
        fn = random.choice(os.listdir(negdir))
        fullpath = os.path.join(negdir, fn)
        y_train[ind, :] = np.array([1, 0])
    else:
        fn = random.choice(os.listdir(posdir))
        fullpath = os.path.join(posdir, fn)
        y_train[ind, :] = np.array([0, 1])
    im = np.flip(cv2.imread(fullpath), axis=2)
    x_train[ind, :, :, :] = misc.imresize(im, (512, 512)) / 255.0

    return


# Generate data in parallel
def generator(batch_size=8):
    manager = multiprocessing.Manager()

    # Infinite loop because data generator
    while True:
        # Declare arrays in shared memory
        mp_x_train = multiprocessing.RawArray(c.c_float, batch_size*512*512*3)
        mp_y_train = multiprocessing.RawArray(c.c_float, batch_size*2)
        x_train = np.frombuffer(mp_x_train, dtype=np.float32).reshape((batch_size, 512, 512, 3))
        y_train = np.frombuffer(mp_y_train, dtype=np.float32).reshape((batch_size, 2))

        # Generate data, randomly selecting positive or negative sample
        jobs = [None] * batch_size
        for i in range(batch_size):
            jobs[i] = multiprocessing.Process(target=generateImage,
                                              args=(x_train, y_train, i))
            jobs[i].start()
        for i in range(batch_size):
            jobs[i].join()

        yield x_train, y_train
