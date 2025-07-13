import argparse
import numpy as np
import sys
import csv
import os
import json
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from Model.data import CIFData, get_train_val_loader, collate_pool
from Model.model import ClusterModel
from Model.training import validate_pre

parser = argparse.ArgumentParser(description='CCSN')
parser.add_argument('--program_name', default='Bandgap_task',
                    help='this is for test (default: hello)')
parser.add_argument('--id_prop_root', default='../../Data_sample/id_prop.xlsx',
                    help='this is the root for id_prop_file')
parser.add_argument('--cif_root', default='../../Data_sample/CIF/',
                    help='this is the root for cif file')
parser.add_argument('--atom_init_root', default='../../Data_sample/atom_init.json',
                    help='this is the root for atom initial file')
parser.add_argument('--fold_info_root', default='../../Data_sample/fold_val.json',
                    help='this is the root for fold information')
# if you want to predict you need to create new fold file for prediction,include all id for prediction in the fold list[0]
parser.add_argument('--prop_name', default='bandgap',
                    help='prop_name')


parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam')
parser.add_argument('--atom-fea-len', default=32, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=32, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-block', default=2, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')


parser.add_argument('--folds', default=1, type=int,
                    metavar='N', help='fold verification')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate')#0.08:min0.59, 0.02:mim0.64, 0.05(megnet16):0.58), 0.08(megnet16):0.563)
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--lr-milestones', default=[350,450], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler')#[300,350,360]]
parser.add_argument('--gamma', default=0.316, type=float,
                    help='decay rate of learning rate')#0.316


parser.add_argument('--use-cuda', default=True,
                    help='want to use CUDA or not')
parser.add_argument('--cuda', default=True,
                    help=' use CUDA or not')

parser.add_argument('--print-freq', '-p', default=30, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers ')
parser.add_argument('--QianYi_root', default='best_Single-cluster_BG_checkpoint_0_.pth.tar',
                    help='this is the root for fold information')
#parser.add_argument('--QianYi_root', default='best_Single-cluster_BG_checkpoint_0_.pth.tar',
                    #help='this is the root for fold information')

args = parser.parse_args(sys.argv[1:])

args.cuda = args.use_cuda and torch.cuda.is_available()
best_mae_error = 1e10


'''###------------------------------------------------------------------------------------------------------------###'''


def main():
    global args, best_mae_error
    random_seed = 123
    random.seed(random_seed)
    print(args.program_name)

    dataset = CIFData(args)
    a = dataset[0]
    print('Loaded data')
    # normalizer = Normalizer(dataset.prop_list)
    normalizer = Normalizer_old(dataset.prop_list)

    with open(args.fold_info_root, 'r') as f:
        fold_info = json.load(f)
    for i in range(len(fold_info)):
        for j in dataset.nan_idx:
            if(j in fold_info[i]):
                fold_info[i].remove(j)

    foldmae = []
    for fold in range(args.folds):
        collate_fn = collate_pool
        train_indices = []
        for i in range(len(fold_info)):
            if i == fold:
                val_indices = fold_info[i]
            else:
                train_indices.extend(fold_info[i])

        train_loader, val_loader = get_train_val_loader(dataset,collate_fn,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.workers,
                                                        val_indices=val_indices,
                                                        train_indices=train_indices)

        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = ClusterModel(orig_atom_fea_len, nbr_fea_len,
                             atom_fea_len=args.atom_fea_len,
                             n_block=args.n_block,
                             h_fea_len=args.h_fea_len,
                             )

        model_name = args.QianYi_root
        print("=> loading model '{}'".format(model_name))
        map_location = torch.device('cpu')
        if (not args.cuda):
            checkpoint = torch.load(model_name, map_location=map_location)
        else:
            checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(model_name, checkpoint['epoch'],
                      checkpoint['best_mae_error']))
        if args.cuda:
            model.cuda()

        criterion = nn.MSELoss()
        if args.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), args.lr,
                                   weight_decay=args.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')
        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                                gamma=args.gamma)


        outfile = 'predict_result_useMC2D.xlsx'
        mae_error = validate_pre(val_loader, model, criterion, normalizer, args, fold, outfile=outfile, test=True)
        # print('mae', mae_error)
        foldmae.append(float(mae_error))


    print(foldmae)
    print('mean of foldmae:', np.array(foldmae).mean())

'''###------------------------------------------------------------------------------------------------------------###'''


class Normalizer(object):
    def __init__(self, tensor):
        # self.min = float(torch.min(tensor))
        # self.max = float(torch.max(tensor))

        new_tensor = []
        for i in range(len(tensor)):
            if(str(tensor[i])!='nan'):
                new_tensor.append(tensor[i])
        tensor = new_tensor
        self.min = min(tensor)
        self.max = max(tensor)
        print('the max value in target list is: ', self.max, ', the min is: ', self.min)
        if(self.min<0):
            self.min_nor=-1
        else:
            self.min_nor=0
        if(self.max>0):
            self.max_nor=1
        else:
            self.max_nor=0

        self.a = (self.max_nor - self.min_nor) / (self.max - self.min)
        self.b = (self.min_nor * self.max - self.min * self.max_nor) / (self.max - self.min)
        #测试归一化结果
        max_test = self.max * self.a + self.b
        min_test = self.min * self.a + self.b
        if(abs(max_test-self.max_nor)<0.00001 and abs(min_test-self.min_nor)<0.00001):
            print('Normalizer had checked')
        else:
            print('something wrong with Normalizer')
            print(max_test,self.max_nor)
            print(min_test,self.min_nor)

    def norm(self, tensor):
        # return tensor * self.a + self.b
        return tensor * self.a + self.b

    def denorm(self, normed_tensor):
        # return (normed_tensor - self.b) / self.a
        return (normed_tensor - self.b) / self.a

    def state_dict(self):
        return {'max': self.max,
                'min': self.min,
                'max_nor': self.max_nor,
                'min_nor': self.min_nor,
                'a': self.a,
                'b': self.b
                }

    def load_state_dict(self, state_dict):
        self.max = state_dict['max']
        self.min = state_dict['min']
        self.max_nor = state_dict['max_nor']
        self.min_nor = state_dict['min_nor']
        self.a = state_dict['a']
        self.b = state_dict['b']

'''###------------------------------------------------------------------------------------------------------------###'''


class Normalizer_old(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        new_tensor = []
        for i in range(len(tensor)):
            if (str(tensor[i]) != 'nan'):
                new_tensor.append(tensor[i])
        tensor = torch.Tensor(new_tensor)
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

if __name__ == '__main__':
    main()