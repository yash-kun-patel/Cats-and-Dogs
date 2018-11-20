import tensorflow as tf
import time
import cv2
import os

start_time = int(time.time())
CATEGORIES = ["Dog", "Cat"]
DATADIR = 'C:\\Users\\Yash Patel\\Desktop\\Tensorflow Tutorials\\P2\\Test Images'
model = tf.keras.models.load_model('64x3-CNN.model')

def prepare(filepath):
    IMG_SIZE = 200
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Convert Image to GRAYSCALE
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


for img in os.listdir(DATADIR):
    #print(img)
    try:
        prediction = model.predict([prepare(os.path.join(DATADIR,img))])
        #print(prediction)
        print(img + " ---> " + CATEGORIES[int(prediction[0][0])])
    except Exception as e:
        pass

print(time.time() - start_time)