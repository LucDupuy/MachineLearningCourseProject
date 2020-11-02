import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# This enables GPU usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#It should print out "cuda:0" if it can detect your gpu correctly
print(device)

# This is known as the "Nerual Networks section" according to the tutorial ---
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
        x = x.view(32, 16*57*103)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#-----------------------------------------------------------------------------

# Displays a series of images, this is only used in code that has been commented out ---
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# --------------------------------------------------------------------------------------

# Loads(and downloads if they haven't been already) all the datasets ---
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = 'data/dataset1'
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('enemy', 'none')
# ---------------------------------------------------------------------



# get some random training images
dataiter = iter(testloader)
images, labels = dataiter.next() # This is loading the data
images = images.to(device) # The loaded data must then be sent to the GPU
labels = labels.to(device) # Send to GPU

# Location of the trained algo
PATH = './cifar_net.pth'

# Loading the neural network code 
net = Net()
net = net.to(device) # Load to GPU
net.load_state_dict(torch.load(PATH))

outputs = net(images).to(device) # Load to GPU

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
        images, labels = data[0].to(device), data[1].to(device) # Load to GPU
        outputs = net(images).to(device) # Load to GPU
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device) # Load to GPU
        outputs = net(images).to(device) # Load to GPU
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i].to(device) # Load to GPU
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

# ---------------------------------------------
