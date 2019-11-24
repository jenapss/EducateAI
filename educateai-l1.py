from __future__ import absolute_import
from __future__ import division
from __future__ import print_function,unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
#import tensorflow_datasets as tfds
import numpy as np
from keras import Input
from keras.utils import np_utils
num_train = 60000 #there are 60000 training examples in MNIST
num_test = 10000 # there are 10000 test examples in MNIST
height,width, depth = 28,28,1
num_classes = 10



mnist_data = keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels) = mnist_data.load_data()

train_images = train_images /255.0
test_images = test_images /255.0

'''
in = Input(shape = (height * width,))
 '''
print(tf.__version__)
print(len(train_labels))
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation = 'relu'),
    keras.layers.Dense(10,activation = 'softmax')
])

model.compile(optimizer = 'adam',
                 loss = 'sparse_categorical_crossentropy',
                 metrics = ['accuracy']
                 )
model.fit(train_images,train_labels,epochs = 10)

test_loss,test_acc = model.evaluate(test_images,test_labels, verbose = 2 )
print("\nTest accuracy : ", test_acc)
predictions = model.predict(test_images)
print(np.argmax(predictions[0]))
print("here")
print(test_labels[0])