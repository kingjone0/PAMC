from torch import nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """LeNet-5 without padding in the first layer.
       This is based on Caffe's implementation of Lenet-5 and is slightly different
       from the vanilla LeNet-5. Note that the first layer does NOT have padding
       and therefore intermediate shapes do not match the official LeNet-5.
       Based on https://github.com/mi-lad/snip/blob/master/train.py
       by Milad Alizadeh.
       """

    def __init__(self, class_num):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=0, bias=True)
        self.conv2 = nn.Conv2d(20, 50, 5, bias=True)
        self.fc3 = nn.Linear(50 * 4 * 4, 500)
        self.fc4 = nn.Linear(500, class_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.fc3(x.view(-1, 50 * 4 * 4)))
        x = self.fc4(x)
        return x


class shared_layers(nn.Module):
    def __init__(self):
        super(shared_layers, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=0, bias=True)  # Adjust the number of input channels if needed
        self.conv2 = nn.Conv2d(20, 50, 5, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        return x


class personalized_layers(nn.Module):
    def __init__(self, num_classes):
        super(personalized_layers, self).__init__()
        self.fc1 = nn.Linear(50 * 4 * 4, 500)  # Ensure the input features match the output of last shared layer
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = x.view(-1, 50 * 4 * 4)  # Flatten the features from the convolutional layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet5_separation(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5_separation, self).__init__()
        self.shared_layers = shared_layers()
        self.personalized_layers = personalized_layers(num_classes)

    def forward(self, x):
        x = self.shared_layers(x)
        x = self.personalized_layers(x)
        return x
