# -*- coding: utf-8 -*-
# 2018-11-15 Started work on predict.py

 # Imports here
import sys
import argparse
import json

import numpy as np

import torch
from torch import nn

import torchvision as tv

#from random import randint 

from PIL import Image

from collections import OrderedDict

from time import time, gmtime, strftime

# Import the helper files
import utilities as utl

#COOKBOOK:
#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

print(utl.get_time_string(), ' Script <predict.py> started.')

def get_input_args(my_argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('imgfile', type = str, help = 'path to the image file to be classified')
    parser.add_argument('network', type = str, help = 'path to the network file')
    parser.add_argument('numclass', type = int, help = 'how many classes should be predicted')
    parser.add_argument('--device', type = str, choices=['cpu', 'cuda:0'], default = 'cpu', help='Device to compute the predictions with')
    parser.add_argument('--namefile', type = str, default = './cat_to_name.json', help = 'path to the JSON file with class names')
    parser.add_argument('--dirvalid', type = str, default = r'./flowers/valid', help = 'path to the folder containing our flower validation images.')
    parser.add_argument('--dirtest', type = str, default = r'./flowers/test', help = 'path to the folder containing our flower test images.')
    my_args = parser.parse_args()
    
    my_return = []
    for any_string in my_argv:
        my_return.append(any_string)
    my_return.append("Arg Count: " + str(len(my_argv)))
    return my_return, my_args
      
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False   
    
def load_network(filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    
    # if XX in checkpoint: For compatibility with my older save files.
    if not('arch' in checkpoint):
        if 'vgg' in filepath:
            checkpoint['arch'] = 'vgg'
        elif 'densenet' in filepath:
            checkpoint['arch'] = 'densenet'
        elif 'alexnet' in filepath:
            checkpoint['arch'] = 'alexnet'
        else:
            print('FATAL ERROR: Seems not to be a known architecture:', filepath)
            sys.exit()
    
    if (checkpoint['arch'] == 'vgg'):
        my_model = tv.models.vgg16(pretrained = True)
        set_parameter_requires_grad(my_model, True)
        inp_units = my_model.classifier[0].in_features
    elif (checkpoint['arch'] == 'densenet'):
        my_model = tv.models.densenet201(pretrained=True)
        set_parameter_requires_grad(my_model, True)
        inp_units = my_model.classifier.in_features
    elif checkpoint['arch'] == 'alexnet':
        my_model = tv.models.alexnet(pretrained=True)
        set_parameter_requires_grad(my_model, True)
        inp_units = my_model.classifier[1].in_features
    else:
        print('FATAL ERROR: No model specified in input file:', filepath)
        sys.exit()

    # if XX in checkpoint: For compatibility with my older save files.
    if 'labels' in checkpoint:        
        my_labels = checkpoint['labels']
    else:
        my_labels = None

    # if XX in checkpoint: For compatibility with my older save files.
    if not('hidden' in checkpoint):
        if checkpoint['arch'] == 'densenet':
            checkpoint['hidden'] = 500
        else:
            checkpoint['hidden'] = 4096

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(inp_units, checkpoint['hidden'], bias = True)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(checkpoint['hidden'], 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    # if XX in checkpoint: For compatibility with my older save files.
    if not('learn_rate' in checkpoint):
        checkpoint['learn_rate'] = 0.001
    
    my_model.classifier = classifier
    my_model.load_state_dict(checkpoint['model_state_dict'])
    #my_model = my_model.cuda()
    my_optimizer = torch.optim.Adam(my_model.classifier.parameters(), lr=checkpoint['learn_rate'],
                                    betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    my_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    my_optimizer.zero_grad()
    #loss_tmp = checkpoint['loss']
    #epc_old = checkpoint['epoch']
    
    return my_model, my_optimizer, checkpoint['loss'], checkpoint['epoch'], my_labels

def process_image_torchvision(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a pytorch tensor
    '''
    # Using the torchvision instead of numpy
    my_transforms = tv.transforms.Compose([tv.transforms.CenterCrop(500),
                                          tv.transforms.Resize(224),
                                          tv.transforms.ToTensor(),
                                          tv.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    image_T = my_transforms(image) 
    image_T = image_T.unsqueeze_(0)
    return image_T

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image.thumbnail((256, 256)) 
    w,h = image.size
    image = image.crop((w/2-112,h/2-112,w/2+112,h/2+112))
    np_image = np.array(image)/255
    
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    #thx2: https://stackoverflow.com/questions/47335033/why-image-numpy-array-transpose-and-its-inverse-change-color-channel
    np_image = np_image.transpose(2,0,1)    
    np_image = np.reshape(np_image, (1, 3, 224, 224))
    #thx2: https://medium.com/@zhang_yang/reshaping-numpy-arrays-newaxis-v-s-reshape-v-s-expand-dims-521289b73c8a
    return np_image

def main():
    start_time = time()
    in_argx, in_args = get_input_args(sys.argv)
    print("\nXXX: ", in_args.imgfile, "\nXXX: ", in_args.network, "\nXXX: ", in_args.numclass, 
          "\nXXX: ", in_args.dirvalid, "\nXXX: ", in_args.dirtest, "\nXXX: ", in_args.device)
    print(in_argx)
    
    model, optimizer, loss_tmp, epc_old, cat_to_name = load_network(in_args.network, in_args.device)
    if (cat_to_name == None):
        with open(in_args.namefile, 'r') as f:
            cat_to_name = json.load(f)
        

    # https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864
    # https://pytorch.org/docs/stable/autograd.html?highlight=variable#variable-deprecated
    #from torch.autograd import Variable
    
    model.eval()
    with torch.no_grad():
        model.to(in_args.device)
        image = Image.open(in_args.imgfile) 

        myinput_np = process_image(image) 
        myinput = torch.FloatTensor(myinput_np) 
        myoutput = model(myinput.to(in_args.device)) 
        catnum = myoutput.data.cpu().numpy().argmax() + 1 
        catname = cat_to_name.get(str(catnum))
       
    print(catnum, catname)
    
    # http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/
    zero_to_one = torch.nn.Softmax(dim=1)
    
    sortval, sortidx = torch.sort(myoutput, descending = True)
    sortval = zero_to_one(sortval)
    sortval_list = sortval[0].tolist()
    sortidx_list = sortidx[0].tolist()

    top_vals = list()
    top_names = list()
    myzip = zip(sortval_list,sortidx_list)
    for cnt, sv in enumerate(myzip):
        print(sv[0], cat_to_name.get(str(sv[1]+1)))
        top_vals.append(sv[0])
        top_names.append(cat_to_name.get(str(sv[1]+1)))
        if cnt >= in_args.numclass:
            break    

    end_time = time()
    tot_time = end_time - start_time
    
    print("\n** Total Elapsed Runtime:", strftime("%H:%M:%S", gmtime(tot_time)))
     

   
if __name__ == "__main__":
    net_dict = dict()
    main()
