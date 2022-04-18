from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
from torch import cuda
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import seaborn as sns
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix

from simple_ann_classifier.ann_classification import Model
from simple_ann_classifier.ann_classification import Test_DiabetesDataset



print(torch.__version__)
device = 'cuda' if cuda.is_available() else 'cpu'
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■ setting ■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
test_dataset = Test_DiabetesDataset()
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=len(test_dataset),
                         shuffle=True)


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■  Model load and Eval■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
model = Model()
model = torch.load('best_model.pt')
model = model.to(device)
model.eval()

test_loader_all = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
for batch_idx, (inputs_all, labels_all) in enumerate(test_loader_all):
    inputs_all, labels_all = inputs_all.to(device=device).float(), labels_all.to(device=device).float()
    y_pred_all = model(inputs_all).to(device)
    y_pred_all = y_pred_all.data.max(1, keepdim=True)[1]

    y_pred_all = y_pred_all.cpu().detach().numpy()
    labels_all = labels_all.cpu().detach().numpy()


plt.figure(2)
confmat = confusion_matrix(y_true=labels_all, y_pred=y_pred_all)
sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues')
plt.title('supervised classifier')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
