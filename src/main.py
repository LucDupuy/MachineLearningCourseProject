import matplotlib.pyplot as plt
import train
from tensorflow.keras import datasets, layers, models

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

# Order pre defined by the existing dataset
classification_arr = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# Building and training the Neural Network
train.train_model(x_train, y_train, x_test, y_test)






"""

Normalize the data to between 0 and 1: Take the images (x_train and x_test) and divide by 255 (max RGB value)

Classification Array Mapping Object to Index -> ["enemy", "teammate"] would set the label of images with enemies as 0 
and images with enemies as 1 

Build the Neural Network
**************************

Convolutional Filters for features (horse has long legs, cat has point ears, plane has wings, truck is bigger then car)
Max Pooling Layer reduces image to essential information

"""
