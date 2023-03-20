import unittest
import torch
import torch.nn as nn

class TestLossFunction(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLossFunction, self).__init__(*args, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        self.input = torch.randn(3, requires_grad=True)
        self.input2 = torch.sigmoid(self.input)
        self.target = torch.empty(3).random_(2) # 
        self.m = nn.Sigmoid()
        self.input3 = self.m(self.input)
        self.loss = nn.BCELoss(reduction='none')
        self.loss2 = nn.BCEWithLogitsLoss(reduction='none')
        print(f"input : ", {self.input})
        print(f"input2 : ", {self.input2})
        print(f"input3 : ", {self.input3})
        output = self.loss(self.input2, self.target)
        print(output)
        output = self.loss2(self.input, self.target)
        print(output)
            
    def test_lossfunction(self):
        with torch.no_grad():
            output = self.loss(self.input2, self.target)
        print(output)
    
    def test_lossfunction2(self):
        with torch.no_grad():
            output = self.loss2(self.input, self.target)
        print(output)
    
    def test_lossfunction3(self):
        torch.manual_seed(77)
        print(f"{'Setting up binary case':-^80}") 
        z = torch.randn(5) # z = f(x)
        y_hat = torch.sigmoid(z) # y_hat = sigmoid(f(x))
        y = torch.tensor([0.,1.,1.,0.,1.]) # ground truth (float type)
        print(f"z = {z}\ny_hat = {y_hat}\ny = {y}\n{'':-^80}")
        # Negative Log-likelihoods
        loss_NLL_scratch = -(y * y_hat.log() + (1 - y) * (1 - y_hat).log())
        print(f"Negative Log-likelihoods\n    {loss_NLL_scratch}")
        # BCELoss from PyTorch
        loss_BCE = self.loss(y_hat,y) # Input : y_hat, y #nn.BCELoss(reduction='none')(y_hat,y) # Input : y_hat, y
        print(f"PyTorch BCELoss\n    {loss_BCE}")
        # BCEWithLogitLoss from PyTorch
        loss_BCEWithLogits = self.loss2(z,y) #nn.BCEWithLogitsLoss(reduction='none')(z,y) # Input : z, y
        print(f"PyTorch BCEWithLogitsLoss\n    {loss_BCEWithLogits}")
        
    def test_lossfunction4(self):
        pred = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], requires_grad=True)
        target = torch.tensor([[0], [2], [1]]).squeeze(1)
        loss = nn.CrossEntropyLoss()
        output = loss(pred, target)
        print(output)
        pred = torch.tensor([[0.1, 0.02, 0.03], [0.2, 0.1, 0.5], [0.2, 0.9, 0.3]], requires_grad=True)
        softmax = nn.Softmax(dim=1)
        pred = softmax(pred)
        print(pred)
        target = torch.tensor([[0], [2], [1]])
        loss = nn.CrossEntropyLoss()
        output = loss(pred, target)
        print(output)
        # hot encoding
        target = torch.zeros(3, 3).scatter_(1, target, 1)
        print(target)
        loss = nn.CrossEntropyLoss()
        output = loss(pred, target)
        print(output)
        

if __name__ == '__main__':
    unittest.main()