# References
# https://github.com/pytorch/examples/blob/master/mnist/main.py
# https://github.com/hunkim/PyTorchZeroToAll/blob/master/09_2_softmax_mnist.py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
from torch import cuda
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

device = 'cuda' if cuda.is_available() else 'cpu'


Epoch_num = 20
batch_size = 100
train_loss_values = np.zeros([Epoch_num - 1])
test_loss_values = np.zeros([Epoch_num - 1])

# ========================================
#           Preparing Dataset
# ========================================
class Train_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/train_freezed.csv',delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]

        self.x_data = from_numpy(xy[:, :-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Test_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/test_freezed.csv',delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, :-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

train_dataset = Train_Dataset()
test_dataset = Test_Dataset()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)


# ========================================
#           Defining a Model
# ========================================
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(10, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 7)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x):
        x1 = self.relu(self.l1(x))
        x2 = self.relu(self.l2(x1))
        x3 = self.l3(x2)
        return x3, x1


# our model
model = Model()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# ========================================
#           Training process
# ========================================
def train(epoch, train_loss_values):
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # get the inputs
        inputs, labels = inputs.to(device=device).float(), labels.to(device=device).float()
        #e Forward pass: Compute predicted y by passing x to the model
        optimizer.zero_grad()
        y_pred, output1 = model(inputs)
        labels = torch.reshape(labels, [-1]).to(device)
        loss = criterion(y_pred, labels.long())
        train_loss +=loss
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        output1 = output1.detach().numpy()
        output1 = pd.DataFrame(output1, columns=names)
        output1.to_csv(CSV_NAME, mode='a', header=False, index = True)
        # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    print(train_loss)
    train_loss /= len(train_loader.dataset)
    train_loss_values[epoch-1] = train_loss
    return train_loss_values


def test(test_loss_values):
    #model.eval()
    test_loss = 0
    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device=device).float(), labels.to(device=device).float()
        y_pred, output1 = model(inputs)
        labels = torch.reshape(labels, [-1]).to(device)
        # sum up batch loss
        loss = criterion(y_pred, labels.long()).item()
        test_loss +=loss
        # get the index of the max
        pred = y_pred.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    test_loss_values[epoch-1] = test_loss

    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')
    return test_loss_values


if __name__ == '__main__':
    since = time.time()
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    CSV_NAME = "test.csv"
    fcsv = open(CSV_NAME, "w")
    names = []
    for i in range(0, 32):
        names.append('node_{}'.format(i + 1))
    # names_ = str(names)[1:-1]
    # fcsv.write(''+ names_)
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

    for epoch in range(1, Epoch_num):
        epoch_start = time.time()
        train_loss_values = train(epoch, train_loss_values)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test_loss_values = test(test_loss_values)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')
    fcsv.close()

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')
    print(train_loss_values)
    print(test_loss_values)
    x_len = np.arange(len(train_loss_values))
    plt.plot( x_len ,test_loss_values, marker = '.',label="Test-set Loss")
    plt.plot( x_len ,train_loss_values,marker = '.', label="Train-set Loss")
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
