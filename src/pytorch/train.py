import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CustomDataset import ImportDataset
from tqdm import tqdm

# This enables GPU usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# It should print out "cuda:0" if it can detect your gpu correctly
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

# -----------------------------------------------------------------------------


# Parameters
batch_size = 32

# Loads(and downloads if they haven't been already) all the datasets ---
all_images_path = r'C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/dataset/images'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = ImportDataset(excel_file='dataset.xlsx', dir=all_images_path, transform=transforms.ToTensor())
# GET RID OF MAGIC NUMBERS IN LINE BELOW

train_set, test_set = torch.utils.data.random_split(dataset, [60128,25810])
trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

"""trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)"""

# ----------------------------------------------------------------------

data = trainloader

net = Net()
net = net.to(device)  # Send to GPU

# The Loss function and Optimizer
# We can change parameters of the algo or change to a different algo completely here
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# The TRAINING of the algo
for epoch in range(50):  # loop over the dataset multiple times
    print("Total Iteration Per Epoch: ", 60128/batch_size)
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
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
# ------------

# Save the algo
PATH = './model.pth'
torch.save(net.state_dict(), PATH)
