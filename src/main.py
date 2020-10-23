import train
from tensorflow.keras import datasets

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

# Order pre defined by the existing dataset
classification_arr = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# Building and training the Neural Network
# train.train_model(x_train, y_train, x_test, y_test)






"""

Normalize the data to between 0 and 1: Take the images (x_train and x_test) and divide by 255 (max RGB value)

Classification Array Mapping Object to Index -> ["enemy", "teammate"] would set the label of images with enemies as 0 
and images with enemies as 1 

Build the Neural Network
**************************

Convolutional Filters for features (horse has long legs, cat has point ears, plane has wings, truck is bigger then car)
-> Multiple layers of neurons, ending with a layer of 2 neurons, representing probability of image being something
-> First layer in input, then are the hidden layers, then the last layer is the output layer
-> "relu activation function": If value is negative, say 0, if positive, return that value (in contrast to sigmoid 
function). This is how the neurons works

-> Max Pooling summarizes (downsamples) the layers to make it easier to work with

Max Pooling Layer reduces image to essential information


Testing Your Model
*******************
After building the neural network, find some test images that the model has not yet seen

prediction = model.predict(np.array([image to test]) / 255)
index = np.argmax(prediction)
print(prediction)

"""


# Cite libraries (ok to use)
# Pytorch


#ffmpeg -i input -vf scale=px:px out.jpg