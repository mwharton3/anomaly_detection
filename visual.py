from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from vis.visualization import visualize_cam
import warnings
warnings.filterwarnings("ignore")


def showAnomalies(model, x_test, y_test, ind):
    # Extract specific image and plot
    testim = x_test[ind, :, :, :].reshape((1, 512, 512, 3))
    grad = misc.imresize(visualize_cam(model, 19, 1, seed_input=testim), (600, 800))
    im = misc.imresize(x_test[ind, :, :, :], (600, 800, 3))
    plt.figure(figsize=(12, 9))
    plt.axis('off')
    plt.imshow(im)
    plt.imshow(grad, 'gray', alpha=0.2);
    plt.show()

    return
