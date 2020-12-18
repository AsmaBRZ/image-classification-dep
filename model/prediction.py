from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,Conv2D,BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda
import h5py
import json
from keras.layers import Dense, Activation, Flatten, Dropout
from keras import regularizers
import numpy as np
from skimage.transform import resize
from PIL import Image
import pathlib
from flask import jsonify 
import tensorflow as tf

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
   
ab = str(pathlib.Path(__file__).parent.absolute())+"/"

WEIGHTS_PATH = ab+'my_model.h5'

def create_model():
    W_DECAY = 0.0001
    IMG_SHAPE = (32, 32, 3)
    NUM_CLASSES = 10
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(W_DECAY), input_shape = IMG_SHAPE))
    model.add(Activation('relu')) 

    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(W_DECAY)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(W_DECAY)))
    model.add(Activation('relu')) 

    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(W_DECAY)))
    model.add(Activation('relu')) 
    
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(W_DECAY)))
    model.add(Activation('relu')) 

    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(W_DECAY)))
    model.add(Activation('relu')) 
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model


model_w = create_model()
model_w.load_weights(WEIGHTS_PATH) 


def predict(data):  
    CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    IMG_SHAPE = (32,32)
    image = Image.open(data)
    image=np.array(image)
    print(image.shape)
    
    image = image.astype('float32')

    image = resize(image, (32, 32), anti_aliasing=True)
    image /= 255

    test_images=[image]
    test_images = np.array(test_images)

    Y_pred_test = model_w.predict(test_images) # Predict probability of image belonging to a class, for each class
    Y_pred_test_classes = np.argmax(Y_pred_test, axis=1) # Class with highest probability from predicted probabilities
    Y_pred_test_max_probas = np.max(Y_pred_test, axis=1) # Highest probability

    # inspect preditions
    label = CIFAR10_CLASSES[Y_pred_test_classes[0]]
    proba = str(round(Y_pred_test_max_probas[0]*100.0,2))
    print(label,proba)
    obj = { 'class': label, 'proba': proba }
    return json.dumps(obj)


    