# References
# https://github.com/pytorch/examples/blob/master/mnist/main.py
# https://github.com/hunkim/PyTorchZeroToAll/blob/master/09_2_softmax_mnist.py
# https://discuss.pytorch.org/t/how-do-i-print-output-of-each-layer-in-sequential/5773/3

from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
from torch import cuda
import numpy as np
import torch
import tensorflow as tf
import time
from pytorch_TTCG import TTCG
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

print(torch.cuda.get_device_name())
print( torch.cuda.is_available())
print(  torch.__version__)

#device = 'cuda' if cuda.is_available() else 'cpu'
device = 'cpu'

Epoch_num = 20
batch_size = 100
train_loss_values = np.zeros([Epoch_num - 1])
test_loss_values = np.zeros([Epoch_num - 1])
scaler = StandardScaler()


class Train_DiabetesDataset(Dataset):
    """ Diabetes dataset."""
    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./NN_data/train_freeze.csv',delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]

        self.x_data = from_numpy(xy[:, :-1])
        # scaler.fit(self.x_data)
        # self.x_data  = scaler.transform(self.x_data)
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Test_DiabetesDataset(Dataset):
    """ Diabetes dataset."""
    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./NN_data/test_freeze.csv',delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, :-1])
        # scaler.fit(self.x_data)
        # self.x_data = scaler.transform(self.x_data)
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

train_dataset = Train_DiabetesDataset()
test_dataset = Test_DiabetesDataset()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)


# To print layer outputs
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here

        # 1. Printing layer output as Tensor form
        print(x)
        # 2. Printing layer output as Numpy form
        print(x.detach().numpy())
        return x


## Method 2 for - Checking Layer outputs
# class Model(nn.Module):
#
#     def __init__(self):
#         """
#         In the constructor we instantiate two nn.Linear module
#         """
#         super(Model, self).__init__()
#         self.l1 = nn.Linear(10, 16)
#         self.l2 = nn.Linear(16, 16)
#         self.l3 = nn.Linear(16, 7)
#
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax()
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#     def forward(self, x):
#         """
#         In the forward function we accept a Variable of input data and we must return
#         a Variable of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Variables.
#         """
#         x = self.relu(self.l1(x))
#         x = self.relu(self.l2(x))
#         return self.l3(x)
# # our model
# model = Model()
# model.to(device)

## Method 1 for - Checking Layer outputs
model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        PrintLayer(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 7),
        )
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

def train(epoch, train_loss_values):
    train_loss = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # get the inputs
        inputs, labels = inputs.to(device=device).float(), labels.to(device=device).float()


        #e Forward pass: Compute predicted y by passing x to the model
        optimizer.zero_grad()
        y_pred = model(inputs)
        labels = torch.reshape(labels, [-1]).to(device)

        loss = criterion(y_pred, labels.long())
        train_loss +=loss
        loss.backward()

        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    print(train_loss)
    train_loss /= len(train_loader.dataset)
    #train_loss_values.append(loss / len(inputs))
    #train_loss_values = np.append(train_loss_values, train_loss)
    train_loss_values[epoch-1] = train_loss
    return train_loss_values


def test(test_loss_values):
    #model.eval()
    test_loss = 0
    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device=device).float(), labels.to(device=device).float()
        y_pred = model(inputs)
        labels = torch.reshape(labels, [-1]).to(device)

        # sum up batch loss
        loss = criterion(y_pred, labels.long()).item()
        test_loss +=loss

        # get the index of the max
        pred = y_pred.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    #test_loss_values.append(test_loss)
    # = np.append(test_loss_values, test_loss)
    test_loss_values[epoch-1] = test_loss

    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')
    return test_loss_values


if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, Epoch_num):
        epoch_start = time.time()
        train_loss_values = train(epoch, train_loss_values)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test_loss_values = test(test_loss_values)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')



    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')
    print(train_loss_values)
    print(test_loss_values)

    x_len = np.arange(len(train_loss_values))

    test_loss_values_mean = np.mean(test_loss_values, axis=0)
    test_standard_dev = np.std(test_loss_values, axis=0)
    train_loss_values_mean = np.mean(train_loss_values, axis=0)
    train_standard_dev = np.std(train_loss_values, axis=0)

    # plt.plot(test_loss_values_mean, train_loss_values_mean )
    # plt.fill_between( test_loss_values_mean - test_standard_dev, test_loss_values_mean+ test_standard_dev , label="Test-set Loss")
    # plt.fill_between( train_loss_values_mean - train_standard_dev, train_loss_values_mean+ train_standard_dev , label="Train-set Loss")

    plt.plot( x_len ,test_loss_values, marker = '.',label="Test-set Loss")
    plt.plot( x_len ,train_loss_values,marker = '.', label="Train-set Loss")
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
