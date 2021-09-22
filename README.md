# PyTorch-1D-NN-Classifier.py

This code is based on the reference codes linked below.  
[reference 1](https://github.com/pytorch/examples/blob/master/mnist/main.py)  [reference 2](https://github.com/hunkim/PyTorchZeroToAll/blob/master/09_2_softmax_mnist.py)  
Purpose of this code is **1-D array data** classification.<br/>
The given data in 'data' directory is simple data for training and testing this code.

# NN_1D_classifier_w_LayerOut_Check.py
[reference 3](https://discuss.pytorch.org/t/how-do-i-print-output-of-each-layer-in-sequential/5773/3)  
This file is same with above file. The only difference is that it's set up so you can watch the outputs of the layer.  
The related code is shown below.  
```python
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
```  
And, this class is applied on the below location.  
```python
model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        PrintLayer(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 7),
        )
model.to(device)
```  
If you run this file, you might obtain below results.  
<img src ="https://user-images.githubusercontent.com/71545160/134338330-e8e07224-6cba-4c9d-b5a5-9f4995d25581.png" width="60%">  
<img src ="https://user-images.githubusercontent.com/71545160/134338939-fe92ac9a-7afc-4d01-83ae-e72cc4e7ad63.png" width="60%">

