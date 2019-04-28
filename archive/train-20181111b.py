# 2018-04-11: Yay! It is beginning!
# 2018-11-06: Deleted conda installation and re-installed, Spyder was broken.
# """"-""-"": CPU Workspaces at Udacity broken since 3 days, Ticket submitted
# 2018-11-10: CPU Workspace seems to work again.


 # Imports here
import sys
import argparse

#import matplotlib as plt
#import seaborn as sb

import numpy as np
#import pandas as pd

import torch
from torch import nn
#from torch import optim
#import torch.nn.functional as F

from torch.optim import lr_scheduler

import torchvision as tv

#import json

#from random import randint 

#from PIL import Image

from collections import OrderedDict

from time import time, gmtime, strftime

# Import the helper files
import myutilities as myu

#COOKBOOK:
#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

print(myu.get_time_string(), ' Script <train.py> started.')

def get_input_args(my_argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, default = r'./flowers/train', help = 'path to the folder containing our flower training images.')
    parser.add_argument('--arch', type = str, default = 'vgg', choices=['vgg', 'densenet', 'resnet'], help = 'ANN model to use.')
    parser.add_argument('--namefile', type = str, default = 'cat_to_name.json', help = 'List of flower names for numbered folders')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Number of epochs to train')
    my_args = parser.parse_args()
    
    my_return = []
    for any_string in my_argv:
        my_return.append(any_string)
    my_return.append("Arg Count: " + str(len(my_argv)))
    return my_return, my_args

# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#set-model-parameters-requires-grad-attribute
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def prep_network(arg_arch, train_dir):
    train_transforms = tv.transforms.Compose([tv.transforms.RandomHorizontalFlip(),
                                              tv.transforms.RandomVerticalFlip(),
                                              tv.transforms.RandomRotation(30),
                                              tv.transforms.RandomCrop(500),
                                              tv.transforms.Resize(224),
                                              tv.transforms.ToTensor(),
                                              tv.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
 
    train_dataset = tv.datasets.ImageFolder(root = train_dir, transform = train_transforms)

    if arg_arch == 'vgg':
        # TODO: Define your transforms for the training, validation, and testing sets
              #Actually the .Normalize parameters are from:
        #https://pytorch.org/docs/master/torchvision/models.html
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
        model = tv.models.vgg16(pretrained=True)
        set_parameter_requires_grad(model, True)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(num_ftrs, 4096, bias = True)),
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(4096, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        optimizer.zero_grad()
        
    elif arg_arch == 'densenet':
        #Actually the .Normalize parameters are from:
        #https://pytorch.org/docs/master/torchvision/models.html
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        model = tv.models.densenet201(pretrained=True)
        set_parameter_requires_grad(model, True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(num_ftrs, 500)),
                ('relu', nn.ReLU()),
                ('drop', nn.Dropout(p=0.2)),
                ('fc2', nn.Linear(500, 102)),
                ('output', nn.LogSoftmax(dim=1))
                ]))
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
        optimizer.zero_grad()

        
    elif arg_arch == 'resnet':
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        model = tv.models.resnet152(pretrained=True)
        set_parameter_requires_grad(model, True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 102)
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer.zero_grad()     
        my_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        net_dict['scheduler'] = my_scheduler
        
    else:
        print('ERROR! [def transforms(arg_arch):] No Transforms and Loaders defined!')
        sys.exit()
            
    net_dict['model'] = model
    net_dict['dataloader'] = dataloader_train
    net_dict['dataset'] = train_dataset
    net_dict['optimizer'] = optimizer
    net_dict['criterion'] = criterion
    

#    return model, dataloader_train, train_dataset, optimizer, criterion
    
def train(num_epochs):
    net_dict['epoc_num'] = num_epochs
    net_dict['model'].cuda()
    net_dict['model'].train()

    print(myu.get_time_string(), 'TRAINING IS STARTING !!!')
    
    for epc in range(net_dict['epoc_num']):
        print(myu.get_time_string(), 'Now training EPOCH #', epc+1, ' of ', net_dict['epoc_num'])
        cnt = 0
        loss_list = []
        if 'scheduler' in net_dict:            
            net_dict['scheduler'].step()
        for cnt, (myinput, mytarget) in enumerate(net_dict['dataloader']):
            mytarget = mytarget.cuda()
            myinput = myinput.cuda()
            net_dict['optimizer'].zero_grad()
        
            myoutput = net_dict['model'](myinput)
            net_dict['loss'] = net_dict['criterion'](myoutput, mytarget)
            
            net_dict['loss'].backward()
            net_dict['optimizer'].step()
    
            loss_list.append(net_dict['loss'].item())
            if cnt % 100 == 0:
                print(cnt, np.mean(loss_list))            
    
    print(myu.get_time_string(), 'TRAINING IS FINISHED !!!')

def save_network(arg_arch):
    state = {
        'epoch': net_dict['epoc_num'],
        'model_state_dict': net_dict['model'].state_dict(),
        'optimizer_state_dict': net_dict['optimizer'].state_dict(),
        'loss': net_dict['loss']
    }
    if 'scheduler' in net_dict:
        state['scheduler_state_dict'] = net_dict['scheduler'].state_dict()
    torch.save(state, './scr_' + myu.get_time_string() + '-' + str(arg_arch) +'_save.pth')


def main():
    start_time = time()
    in_argx, in_args = get_input_args(sys.argv)
    print("Directory:", in_args.dir, "\nCNN Type:", in_args.arch, "\nJSON file with Flower names:", in_args.namefile,
          "\nEpochs: ", in_args.epochs)
    print(in_argx)
    
    prep_network(in_args.arch, in_args.dir)
    
    train(in_args.epochs)
    
    save_network(in_args.arch)
            
    end_time = time()
    tot_time = end_time - start_time
    
    print("\n** Total Elapsed Runtime:", strftime("%H:%M:%S", gmtime(tot_time)))
    
    #torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    #Initialize the main dictionary
    net_dict = dict()
#            'dataloader' : None,
#            'dataset' : None,
#            'optimizer' : None,
#            'criterion' : None,
#            'epoc_num' : None
#            }
    main()
