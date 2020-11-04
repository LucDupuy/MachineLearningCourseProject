import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from CustomDataset import ImportDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*57*103, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(16, 16*57*103)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main(train_spreadsheet_path, train_images_path):
    batch_size = 16
    classes = ["enemy", "none"]

    train_set = ImportDataset(excel_file=train_spreadsheet_path, dir=train_images_path, transform=transforms.ToTensor())

    trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    print(train_set.__len__())

    train(trainloader, batch_size)



def train(trainloader, batch_size):
    data = trainloader

    net = Net()
    net = net.to(device)  # Send to GPU

    # The Loss function and Optimizer
    # We can change parameters of the algo or change to a different algo completely here
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # The TRAINING of the algo
    for epoch in range(1):  # loop over the dataset multiple times
        print("Total Iteration Per Epoch: ", 85936 / batch_size)
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
            if i % 50 == 0:  # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    # ------------

    # Save the algo
    PATH = './model.pth'
    torch.save(net.state_dict(), PATH)


# Displays a series of images, this is only used in code that has been commented out ---
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test(testloader, classes):
    # get some random training images
    dataiter = iter(testloader)
    images, labels = dataiter.next()  # This is loading the data
    images = images.to(device)  # The loaded data must then be sent to the GPU
    labels = labels.to(device)  # Send to GPU

    # Location of the trained algo
    PATH = './model.pth'

    # Loading the neural network code
    net = Net()
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
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  # Load to GPU
            outputs = net(images).to(device)  # Load to GPU
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  # Load to GPU
            outputs = net(images).to(device)  # Load to GPU
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i].to(device)  # Load to GPU
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    # ---------------------------------------------


if __name__ == '__main__':
    main(train_spreadsheet_path='C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/training_data/train_set.xlsx',
         train_images_path='C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/training_data/images')
