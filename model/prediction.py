import tensorflow as tf
import h5py
import json
import numpy as np
from PIL import Image
from skimage.transform import resize
from flask import jsonify 
import os

model_w = None


def predict(data):  
    global model_w

    if model_w is None:
        txt=os.path.abspath(__file__)
        x = txt.split("/", 3)
        my_path="/"+x[1]+"/"+x[2]+"/my_model"
        print("************************************************",my_path)
        my_file="/"+x[1]+"/"+x[2]+"/my_model.h5"
        print("************************************************",my_file)
        model_w = tf.keras.models.load_model(my_path)
        model_w.load_weights(my_file) 

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
