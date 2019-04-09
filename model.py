from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import keras.backend as K
import warnings
warnings.filterwarnings("ignore")


# Custom CNN
def CAMCNN(input_size):
    inputs = Input(input_size)

    # Convolution sequence
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)


    avgpool = AveragePooling2D((32, 32))(conv5)
    flat = Flatten()(avgpool)
    output = Dense(2, activation='softmax')(flat)

    model = Model(input=inputs, output=output)

    return model


# Generate VGG with pre-trained weights
def generateVGG16():
    # Get base VGG16
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(512,512,3))
    vgg_model.layers.pop()

    # Terminate to support CAM
    x = vgg_model.layers[-1].output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs=vgg_model.input, outputs=x)

    # Freeze layers
    for layer in vgg_model.layers:
        layer.trainable = False

    # Unfreeze the last few
    for layer in vgg_model.layers[15:]:
        layer.trainable = True

    return model
