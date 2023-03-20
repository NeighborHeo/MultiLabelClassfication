import unittest
import torch

class TestAccuracy(unittest.TestCase):
    
    def test_1():
        target = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
        output = torch.tensor([0.1, 0.4, 0.35, 0.8], dtype=torch.float32)
        acc = torch.sum(torch.eq(target, output > 0.5).float()) / target.shape[0]
        print(acc)

if __name__ == '__main__':
    unittest.main()