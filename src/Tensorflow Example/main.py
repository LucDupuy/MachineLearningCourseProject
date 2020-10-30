import train
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets, models


def main():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train, x_test = x_train / 255, x_test / 255

    # Order pre defined by the existing dataset
    classification_arr = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

    # train_model(x_train, y_train, x_test, y_test)
    prediction = predict()
    print(f'Prediction is {classification_arr[prediction]}')


def train_model(x_train, y_train, x_test, y_test):
    train.train_model(x_train, y_train, x_test, y_test)


def predict():
    model = models.load_model('image_classifier')

    img = cv.imread("car.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.imshow(img, cmap=plt.cm.binary)

    prediction = model.predict(np.array(img) / 255)
    index = np.argmax(prediction)

    return index

if __name__ == '__main__':
    main()