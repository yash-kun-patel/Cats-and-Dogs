import numpy as np
import cv2
import os
import tensorflow as tf


IMG_SIZE = 200

# We need to normalize all images to the same shape so that there are no such errors and confusions

training_data = []
DATADIR = 'C:\\Users\\Yash Patel\\Desktop\\Tensorflow Tutorials\\P2\\PetImages'    # Path to the set of images
CATEGORIES = ["Dog","Cat"]     #Two categories cats and dogs

def create_training_data():
    i = 10
    for category in CATEGORIES:
        if i < 100:
            i += 1
            path = os.path.join(DATADIR,category)            #Path to cats or dogs directory
        #     print(path)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                # os.listdir coverts all the names of the files in the directory into a list
                try:
                    img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)    # Convert Image to GRAYSCALE
                    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                    training_data.append([new_array,class_num])
                except Exception as e:
                    pass
        i = 0

create_training_data()

print(len(training_data))
print(training_data[0])

import random

random.shuffle(training_data)
for samples in training_data[:10]:
    print(samples[1])

X = []
y = []
for features,labels in training_data:
    X.append(features)
    y.append(labels)
X = np.array(X)
print(X.shape)
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print(X.shape)

#     Export the data so that it is not to be recrunched again and again
#     Export could be done by np.save

import pickle

pickle_out = open("x.pickle",'wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle",'wb')
pickle.dump(y,pickle_out)
pickle_out.close()

#     Import the loaded data

pickle_in = open("x.pickle",'rb')
X = pickle.load(pickle_in)

# Trying NumPy save feature

# from tempfile import TemporaryFile
# outfile = open("X_numpy_save.npy","wb")
# np.save(outfile,X)
# X_train = np.load("X_numpy_save.npy")