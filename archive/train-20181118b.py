# -*- coding: utf-8 -*-

# 2018-04-11: Yay! It is beginning!
# 2018-11-06: Deleted conda installation and re-installed, Spyder was broken.
# """"-""-"": CPU Workspaces at Udacity broken since 3 days, Ticket submitted
# 2018-11-10: CPU Workspace seems to work again.
# 2018-11-18: Began some convention reformatting (Spyder Code Analysis)


 # Imports here
import sys
import argparse

from collections import OrderedDict
import json
from time import time, gmtime, strftime

#import matplotlib as plt
#import seaborn as sb

import numpy as np
#import pandas as pd

import torch
from torch import nn
#from torch import optim
#import torch.nn.functional as F

#from torch.optim import lr_scheduler

import torchvision as tv



#from random import randint 

#from PIL import Image



# Import the helper files
import utilities as utl

#COOKBOOK:
#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

print(utl.get_time_string(), ' Script <train.py> started.')

def get_input_args(my_argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('arch', type=str, choices=['vgg', 'densenet', 'alexnet'], help='ANN model to use.')
    parser.add_argument('device', type=str, choices=['cpu', 'cuda'], help='Type of computation device to use.')
    parser.add_argument('--dir', type=str, default=r'./flowers/train', help='path to the folder containing our flower training images.')
    parser.add_argument('--vdir', type=str, default=r'./flowers/valid', help='path to the folder containing our flower validation images.')
    parser.add_argument('--namefile', type=str, default='cat_to_name.json', help='List of flower names for numbered folders')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--savepath', type=str, default=r'G:\Udacity_aipnd_nnsaves', help='Path to the save directory')
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

def prep_network(arg_arch, mydevice, train_dir, valid_dir, cat_name_file):
    with open(cat_name_file, 'r') as f:
        cat_to_name = json.load(f)
    
    train_transforms = tv.transforms.Compose([tv.transforms.RandomHorizontalFlip(),
                                              tv.transforms.RandomVerticalFlip(),
                                              tv.transforms.RandomRotation(30),
                                              tv.transforms.RandomCrop(500),
                                              tv.transforms.Resize(224),
                                              tv.transforms.ToTensor(),
                                              tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    valid_transforms = tv.transforms.Compose([tv.transforms.CenterCrop(500),
                                              tv.transforms.Resize(224),
                                              tv.transforms.ToTensor(),
                                              tv.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
   
    train_dataset = tv.datasets.ImageFolder(root = train_dir, transform = train_transforms)
    valid_dataset = tv.datasets.ImageFolder(root = valid_dir, transform = valid_transforms)
    dataloader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

    if arg_arch == 'vgg':
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        model = tv.models.vgg16(pretrained=True)
        set_parameter_requires_grad(model, True)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(num_ftrs, 4096, bias = True)),
                ('relu', nn.ReLU()),
                ('drop', nn.Dropout(p=0.2)),
                ('fc2', nn.Linear(4096, 102)),
                ('output', nn.LogSoftmax(dim=1))
                ]))
        model.to(mydevice)
        criterion = nn.CrossEntropyLoss()
        criterion.to(mydevice)
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
        model.to(mydevice)
        criterion = nn.CrossEntropyLoss()
        criterion.to(mydevice)
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
        optimizer.zero_grad()

        
    elif arg_arch == 'alexnet':
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        model = tv.models.alexnet(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        set_parameter_requires_grad(model, True)
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(num_ftrs, 4096, bias = True)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(4096, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.to(mydevice)
        criterion = nn.CrossEntropyLoss()
        criterion.to(mydevice)
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        optimizer.zero_grad()     
        
    else:
        print('ERROR! [def transforms(arg_arch):] No Transforms and Loaders defined!')
        sys.exit()
            
    net_dict['labels'] = cat_to_name
    net_dict['arch'] = arg_arch
    net_dict['model'] = model
    net_dict['dataloader'] = dataloader_train
    net_dict['dataset'] = train_dataset
    net_dict['optimizer'] = optimizer
    net_dict['criterion'] = criterion
    
    return dataloader_valid
    

def val_stats_alpha(model, val_loader, criterion, mydevice):
    val_loss = 0
    val_accur = list()
 
    
    with torch.no_grad():
        for img, lab in val_loader:
            img = img.to(mydevice)
            lab = lab.to(mydevice)
            val_out = model.forward(img)
            val_loss += criterion(val_out, lab).item()
            sft_max = torch.exp(val_out)
            val_eql = (lab.data == sft_max.max(dim=1)[1])
            val_accur.append(val_eql.type(torch.float).mean().cpu())
    return val_loss, np.mean(np.array(val_accur))
    
def val_stats_beta(model, val_loader, criterion, mydevice):
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for(images, labels) in val_loader:  #dataloader_test: #dataloader_valid:
            images = images.to(mydevice)
            labels = labels.to(mydevice)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return val_loss, (correct / total)
    
def train(num_epochs, mydevice, dataloader_valid):
    net_dict['epoc_num'] = num_epochs
    net_dict['model'].cuda()
    net_dict['model'].train()

    print(utl.get_time_string(), 'TRAINING IS STARTING !!!')
    
    for epc in range(net_dict['epoc_num']):
        print(utl.get_time_string(), 'Now training EPOCH #', epc+1, ' of ', net_dict['epoc_num'])
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
                print('Batch #', cnt, ' Training loss: ', np.mean(loss_list))            
        net_dict['model'].eval()
        loss_alpha, accuracy_alpha = val_stats_alpha(net_dict['model'], dataloader_valid, net_dict['criterion'], mydevice)
        print('val_stats_alpha', loss_alpha, accuracy_alpha)
        loss_beta, accuracy_beta = val_stats_beta(net_dict['model'], dataloader_valid, net_dict['criterion'], mydevice)
        print('val_stats_beta', loss_beta, accuracy_beta)
        net_dict['model'].train()
    
    
    
    print(utl.get_time_string(), 'TRAINING IS FINISHED !!!')
    

def save_network(arg_arch, arg_savepath):
    state = {
            'labels' : net_dict['labels'],
            'epoch': net_dict['epoc_num'],
            'arch' : net_dict['arch'],            
            'model_state_dict': net_dict['model'].state_dict(),
            'optimizer_state_dict': net_dict['optimizer'].state_dict(),
            'loss': net_dict['loss']
    }
    if 'scheduler' in net_dict:
        state['scheduler_state_dict'] = net_dict['scheduler'].state_dict()
    torch.save(state, arg_savepath + '/scr_' + utl.get_time_string() + '-' + str(arg_arch) +'_save.pth')


def main():
    start_time = time()
    in_argx, in_args = get_input_args(sys.argv)
    print("Directory:", in_args.dir, "\nCNN Type:", in_args.arch, "\nJSON file with Flower names:", in_args.namefile, 
          "\nDevice: ", in_args.device, "\nValidation Directory: ", in_args.vdir, "\nEpochs: ", in_args.epochs)
    print(in_argx)
    
    validation_data_loader = prep_network(in_args.arch, in_args.device, in_args.dir, in_args.vdir, in_args.namefile)
    
    train(in_args.epochs, in_args.device, validation_data_loader)
    
    save_network(in_args.arch, in_args.savepath)
            
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
