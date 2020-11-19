import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
import os
import matplotlib.pyplot  as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv
from CustomDataset import ImportDataset

load_dotenv()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, input_size, num_class, batch_size):
        self.input_size = input_size
        self.num_class = num_class
        self.batch_size = batch_size

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, self.batch_size, 5)

        # First param is input_size of the image, second is number of nodes (input sample, output sample)
        self.fc1 = nn.Linear(self.input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        # Output is number of classes
        self.fc3 = nn.Linear(84, self.num_class)

    """
    Maps the input tensor to a prediction output tensor
    """

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(self.batch_size, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


batch_sizes = [3, 4, 5, 10, 16, 32, 64]

num_classes = 2
learning_rates = 0.001
momentum = 0.9
num_print_loss = 1000
epochs = 1


def main(train_spreadsheet_path, train_images_path, test_spreadsheet_path, test_images_path, valid_images_path,
         valid_spreadsheet_path):
    classes = ["none", "enemy"]

    for i in range(len(batch_sizes)):
        # Shape for x.view's second parameter has to be [batch size, batch size * constant to satisfy error (cant figure
        # out what "input size of X" refers to]
        input_size = batch_sizes[i] * 4389

        train_set = ImportDataset(excel_file=train_spreadsheet_path, dir=train_images_path,
                                  transform=transforms.ToTensor())
        trainloader = DataLoader(dataset=train_set, batch_size=batch_sizes[i], shuffle=True)

        valid_set = ImportDataset(excel_file=valid_spreadsheet_path, dir=valid_images_path,
                                  transform=transforms.ToTensor())
        validloader = DataLoader(dataset=valid_set, batch_size=batch_sizes[i], shuffle=True)

        print("Training Beginning: \n--------------------------------------")

        train(trainloader, train_size=train_set.__len__(), batch=batch_sizes[i], in_size=input_size)

        validate(validloader, test_size=valid_set.__len__(), batch=batch_sizes[i], in_size=input_size)




def train(trainloader, train_size, batch, in_size):
    tmp_batch = batch
    total_iterations = math.floor(train_size / batch)
    stopping_val = total_iterations - 100
    print(f"Batch Size: {batch}\n")
    net = Net(input_size=in_size, num_class=num_classes, batch_size=batch).to(device)

    # The Loss function and Optimizer
    # We can change parameters of the algo or change to a different algo completely here
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # The TRAINING of the algo
    for epoch in range(epochs):  # loop over the dataset multiple times
        print("Total Iterations Per Epoch: ", total_iterations)
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)  # Also send to GPU
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).to(device)  # Send to GPU
            loss = criterion(outputs, labels).to(device)  # Send to GPU
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # print statistics
            if i % num_print_loss == 0 and i != 1:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / num_print_loss))

                running_loss = 0.0

            # Stop on a whole number of iterations
            if i >= stopping_val and i % tmp_batch == 0:
                break
            else:
                continue

    print('-------------------------------------'
          '\nFinished Training')
    # ------------

    # Save the algo
    PATH = f'./models/model{batch}.pth'
    torch.save(net.state_dict(), PATH)


# Displays a series of images, this is only used in code that has been commented out ---
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def validate(validloader, test_size, batch, in_size):
    tmp_batch = batch
    total_iterations = math.floor(test_size / batch)
    stopping_val = total_iterations - 100
    print("Validation Beginning\n--------------------------------------\n")

    # get some random training images
    dataiter = iter(validloader)

    images, labels = dataiter.next()  # This is loading the data
    images = images.to(device)  # The loaded data must then be sent to the GPU
    labels = labels.to(device)  # Send to GPU

    # Location of the trained algo
    PATH = f'./models/model{batch}.pth'

    # Loading the neural network code
    net = Net(input_size=in_size, num_class=num_classes, batch_size=batch).to(device)
    net = net.to(device)  # Load to GPU
    net.load_state_dict(torch.load(PATH))

    outputs = net(images).to(device)  # Load to GPU

    _, predicted = torch.max(outputs, 1)

    # the following prints one of the tests done, uncomment if you want to see one
    """
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    imshow(torchvision.utils.make_grid(images))
    """

    # Prints out the total accuracy of the algo ---
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(validloader):
            images, labels = data[0].to(device), data[1].to(device)  # Load to GPU
            outputs = net(images).to(device)  # Load to GPU
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i >= stopping_val and i % tmp_batch == 0:
                break
            else:
                continue

    # total_accuracy = 'Accuracy of the network on the test images: %.5f %% \n' % (100 * correct / total)
    total_accuracy = 100 * correct / total

    # Accuracy of each individual class
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for j, data in enumerate(validloader):
            images, labels = data[0].to(device), data[1].to(device)  # Load to GPU
            outputs = net(images).to(device)  # Load to GPU
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i].to(device)  # Load to GPU
                class_correct[label] += c[i].item()
                class_total[label] += 1
            if j >= stopping_val and j % tmp_batch == 0:
                break
            else:
                continue

    class_accuracy = []
    for i in range(num_classes):
        class_accuracy.append(100 * class_correct[i] / class_total[i])

    none_acc = class_accuracy[0]
    enemy_acc = class_accuracy[1]

    createResultsSpreadsheet(network_acc=total_accuracy, none_acc=none_acc, enemy_acc=enemy_acc,
                             batch_size=batch, epoch=epochs, lr=learning_rate)

    # file = open(f"./results/Results{batch}-{epochs}.txt", 'a')
    # file.write(f"Batch Size: {batch}\n")
    # file.write(total_accuracy)
    # file.write(class_accuracy[0])
    # file.write(class_accuracy[1])
    # file.close()

    # ---------------------------------------------


def createResultsSpreadsheet(network_acc, none_acc, enemy_acc, batch_size, epoch, lr):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        results_file = "pytorch/results/New Dataset/Results.xlsx"

        new_row = {'Batch Size': batch_size, 'Epochs': epoch, 'Learning Rate': lr,
                                'Accuracy of Network': network_acc, 'Accuracy of None': none_acc,
                                'Accuracy of Enemy':enemy_acc}

        if os.path.isfile(results_file):
            df = pd.read_excel(results_file)
            df = df.append(new_row, ignore_index=True)


            df.to_excel('C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/Results/New Dataset/Results.xlsx',
                             header=True, index=False)

        else:
            new_df = pd.DataFrame(data=new_row)
            new_df.to_excel('C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/Results/New Dataset/Results.xlsx',
                             header=True, index=False)


if __name__ == '__main__':
    """main(
        train_spreadsheet_path=os.getenv("TRAIN_SPREADSHEET"),
        train_images_path=os.getenv("TRAIN_IMAGES"),
        test_spreadsheet_path=os.getenv("TEST_SPREADSHEET"),
        test_images_path=os.getenv("TEST_IMAGES"))"""
    main(
        train_spreadsheet_path='C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/training_data/train_set.xlsx',
        train_images_path='C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/training_data/images',
        test_spreadsheet_path='C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/testing_data/test_set.xlsx',
        test_images_path='C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/testing_data/images',
        valid_spreadsheet_path='C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/valid_data/valid_set.xlsx',
        valid_images_path='C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/valid_data/images'
    )
