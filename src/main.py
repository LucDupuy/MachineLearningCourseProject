import matplotlib.pyplot as plt
from tensorflow.keras import datasets

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

print(y_test)




"""

Classification Array Mapping Object to Index -> ["enemy", "teammate"] would set the label of images with enemies as 0 
and images with enemies as 1 

"""