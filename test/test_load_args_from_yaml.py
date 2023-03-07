'''test_load_args_from_yaml.py'''

import argparse
import os
import sys
import yaml

def load_args_from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        # args = argparse.Namespace()
        # args_dict = yaml.safe_load(f)
        # args.__dict__.update(args_dict)
        
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        yaml_args = yaml.safe_load(f)
        parser.set_defaults(**yaml_args)
        parser.add_argument('--model_name', default='', type=str, help='model name')
        parser.add_argument('--model_index', default=0, type=int, help='model index')
        parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
        # yaml + argparse
        print(parser.parse_args())
        return parser.parse_args()

def test_load_args_from_yaml():
    args = load_args_from_yaml('test_args.yaml')
    assert args.model_name == 'vit_base_patch16_224'
    assert args.pretrained == True
    assert args.resume == False
    assert args.batch_size == 128
    assert args.lr == 0.1
    assert args.epochs == 200
    assert args.weight_decay == 5e-4
    assert args.momentum == 0.9
    assert args.print_freq == 10
    assert args.save_freq == 10
    assert args.save_path == 'checkpoint'
    assert args.data_path == '/home/andrew/data/cifar10'
    assert args.dataset == 'cifar10'
    assert args.num_classes == 10
    assert args.num_workers == 2
    assert args.seed == 1
    assert args.device == 'cuda'
    assert args.gpu_id == '0'
    assert args.log_path == 'log.txt'
    assert args.log_freq == 10
    assert args.log_to_file == True
    assert args.log_to_comet == True
    return
    
def main():
    args = load_args_from_yaml('/home/suncheol/code/FedTest/pytorch-models/test/test_args.yaml')
    print(args)

if __name__ == '__main__':
    main()
    