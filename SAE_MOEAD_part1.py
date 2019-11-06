# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import copy, time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from GA_pop_repair_v2 import *
from model import AutoEncoder_total
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)

program_start_time = time.time()
# ============= Some parameters and constants ==============
TEST_RATIO = 0.95    
TRAIN_VALID_RATIO = 0.6    

IND_FLOAT_LENGTH = 225
IND_BOOL_LENGTH = 224 + 2

MAX_HIDDEN_LAYER_NUM = 3 

# San Francisco2 
TYPES = 5   
HEIGHT = 1800   
WIDTH = 1380    

FEATURE_DIM = 9     
WHOLE_PIC_BATCH_SIZE = 500000   
TEST_SET_BATCH_SIZE = 500000   
MAX_ITERA_TIMES = 5 
PC = 0.9    
PM = 0.1    
DELTA = 0.9     
NR = 2      
VALID_FREQ = 5    
VALID_PATIENCE = 15     
MAX_EPOCH = 1000000     

EPOCH_CLF = 10 
LR = 0.001
LR_FINAL = 0.001

EP = []
EP_FV = []

weight_list = []    
bias_list = []      
activation_func_list = []     
hidden_dim_list = []

# ==================== load data ========================
data_label_one = pd.read_csv('E:\\PythonProject\\StackedAE_MOEAD\\SanFrancisco2\\filtered_with_7x7_refined_Lee\\data_label_one_5x5_SanFrancisco2.csv', header=None)
data_label_two = pd.read_csv('E:\\PythonProject\\StackedAE_MOEAD\\SanFrancisco2\\filtered_with_7x7_refined_Lee\\data_label_two_5x5_SanFrancisco2.csv', header=None)
data_label_three = pd.read_csv('E:\\PythonProject\\StackedAE_MOEAD\\SanFrancisco2\\filtered_with_7x7_refined_Lee\\data_label_three_5x5_SanFrancisco2.csv', header=None)
data_label_four = pd.read_csv('E:\\PythonProject\\StackedAE_MOEAD\\SanFrancisco2\\filtered_with_7x7_refined_Lee\\data_label_four_5x5_SanFrancisco2.csv', header=None)
data_label_five = pd.read_csv('E:\\PythonProject\\StackedAE_MOEAD\\SanFrancisco2\\filtered_with_7x7_refined_Lee\\data_label_five_5x5_SanFrancisco2.csv', header=None)









