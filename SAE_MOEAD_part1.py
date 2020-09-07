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

MAX_HIDDEN_LAYER_NUM = 4 

# San Francisco2 
TYPES = 5   
HEIGHT = 1800   
WIDTH = 1380    

FEATURE_DIM = 9     
WHOLE_PIC_BATCH_SIZE = 500000  # used when predict the whole pic 
TEST_SET_BATCH_SIZE = 500000   # used when predict the test set 
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

print(data_label_one.values)
print('data_label_one.values.shape: ', data_label_one.values.shape)

label_one = np.ones((data_label_one.values.shape[0], 1))
label_two = 2 * np.ones((data_label_two.values.shape[0], 1))
label_three = 3 * np.ones((data_label_three.values.shape[0], 1))
label_four = 4 * np.ones((data_label_four.values.shape[0], 1))
label_five = 5 * np.ones((data_label_five.values.shape[0], 1))


# ============== train/val/test set split ===============
X_train_val_one, X_test_one, y_train_val_one, y_test_one = \
    train_test_split(data_label_one.values, label_one, test_size=TEST_RATIO, random_state=4)
X_train_val_two, X_test_two, y_train_val_two, y_test_two = \
    train_test_split(data_label_two.values, label_two, test_size=TEST_RATIO, random_state=4)
X_train_val_three, X_test_three, y_train_val_three, y_test_three = \
    train_test_split(data_label_three.values, label_three, test_size=TEST_RATIO, random_state=4)
X_train_val_four, X_test_four, y_train_val_four, y_test_four = \
    train_test_split(data_label_four.values, label_four, test_size=TEST_RATIO, random_state=4)
X_train_val_five, X_test_five, y_train_val_five, y_test_five = \
    train_test_split(data_label_five.values, label_five, test_size=TEST_RATIO, random_state=4)

X_test = np.vstack([X_test_one,X_test_two,X_test_three,X_test_four,X_test_five])    
y_test = np.vstack([y_test_one,y_test_two,y_test_three,y_test_four,y_test_five])    

del data_label_one, data_label_two, data_label_three, data_label_four, data_label_five


X_train_one, X_val_one, y_train_one, y_val_one = \
    train_test_split(X_train_val_one, y_train_val_one, test_size=TRAIN_VALID_RATIO)
X_train_two, X_val_two, y_train_two, y_val_two = \
    train_test_split(X_train_val_two, y_train_val_two, test_size=TRAIN_VALID_RATIO)
X_train_three, X_val_three, y_train_three, y_val_three = \
    train_test_split(X_train_val_three, y_train_val_three, test_size=TRAIN_VALID_RATIO)
X_train_four, X_val_four, y_train_four, y_val_four = \
    train_test_split(X_train_val_four, y_train_val_four, test_size=TRAIN_VALID_RATIO)
X_train_five, X_val_five, y_train_five, y_val_five = \
    train_test_split(X_train_val_five, y_train_val_five, test_size=TRAIN_VALID_RATIO)

X_train = np.vstack([X_train_one,X_train_two,X_train_three,X_train_four,X_train_five])   
y_train = np.vstack([y_train_one,y_train_two,y_train_three,y_train_four,y_train_five])   
X_val = np.vstack([X_val_one,X_val_two,X_val_three,X_val_four,X_val_five])    
y_val = np.vstack([y_val_one,y_val_two,y_val_three,y_val_four,y_val_five])    

del X_train_val_one, X_train_val_two, X_train_val_three, X_train_val_four, X_train_val_five


# ============== z-score ===========
scaler = preprocessing.StandardScaler().fit(X_train)
scaler.transform(X_train)
scaler.transform(X_val)
scaler.transform(X_test)

MOEAD_start_time_0 = time.time()
# ================== load the weight of MOEA/D-ACD ====================
weight_vectors = pd.read_csv("E:\\PythonProject\\StackedAE_MOEAD\\weight_50.csv", header=None)
weight_vec_distance = np.zeros((weight_vectors.shape[0], weight_vectors.shape[0]))
direction_vectors = direction_vector_cal(weight_vectors.values)
theta_min, theta_max, theta = Tchebycheff_MOEAD_ACD(direction_vectors)

POP_SIZE = weight_vectors.shape[0] 
OBJ_NUM = weight_vectors.shape[1]

for i in range(0, weight_vectors.shape[0]-1):
    for j in range(i+1, weight_vectors.shape[0]):
        weight_vec_distance[i, j] = np.linalg.norm(weight_vectors.values[j,:]-weight_vectors.values[i,:])
        weight_vec_distance[j, i] = copy.deepcopy(weight_vec_distance[i, j])

B = np.zeros((weight_vectors.shape[0], T), dtype=int) 

for i in range(weight_vectors.shape[0]):
    neigh_index = np.argsort(weight_vec_distance[i])
    B[i] = copy.deepcopy(neigh_index[0:T])
print("B: ", B)

P_whole = np.zeros([POP_SIZE, POP_SIZE], dtype=int)
for i in range(P_whole.shape[0]):
    for j in range(P_whole.shape[0]):
        P_whole[i, j] = j
print('P_whole:\n', P_whole)
print('P_whole.shape: ', P_whole.shape)

ind_float = np.random.random((weight_vectors.shape[0], IND_FLOAT_LENGTH))  
ind_bool = np.random.randint(0, 2, (weight_vectors.shape[0], IND_BOOL_LENGTH))
pop = np.hstack((ind_float, ind_bool))    

pop = ind_repair(pop, pop.shape[1]-2, pop.shape[1]-1)

FV = np.zeros((POP_SIZE, OBJ_NUM)) 
FV_normalize = np.zeros((POP_SIZE, OBJ_NUM))   
Z = np.zeros((1, OBJ_NUM))
decoder_bias = []

class AutoEncoder(nn.Module):
    
    def __init__(self, weight, bias, activation_func, input_dim, hidden_dim, TYPES):
        
        super(AutoEncoder, self).__init__()

        AE_encoder = nn.Linear(input_dim, hidden_dim) 
        AE_decoder = nn.Linear(hidden_dim, input_dim)

        if activation_func == 0:
            self.encoder = nn.Sequential(AE_encoder, nn.Sigmoid())

        if activation_func == 1:
            self.encoder = nn.Sequential(AE_encoder, nn.Tanh())

        if activation_func == 2:
            self.encoder = nn.Sequential(AE_encoder, nn.ReLU())

        weight_T = nn.Parameter(torch.FloatTensor(weight.T))
        weight = nn.Parameter(torch.FloatTensor(weight))
       
        AE_encoder.weight = weight
        bias = nn.Parameter(torch.FloatTensor(bias))
        AE_encoder.bias = bias
        AE_decoder.weight = weight_T      
        self.decoder = nn.Sequential(AE_decoder)

        self.clf = nn.Linear(hidden_dim, TYPES)

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def classifier(self, encoded):
        return self.clf(encoded)

def reconstr_error(input, decoded, BATCH_SIZE):
    return torch.sum(torch.pow((input-decoded), 2)) / BATCH_SIZE

def CCR(predict, ground_truth):
    loss = nn.CrossEntropyLoss()
    return loss(predict, ground_truth)

net_arch_list = []     

X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)

ones_train = np.ones(y_train.shape)   
ones_val = np.ones(y_val.shape)   
ones_test = np.ones(y_test.shape)   

y_train = torch.LongTensor(y_train-ones_train)
y_train = torch.transpose(y_train, 0, 1)
y_train = y_train.squeeze(0)       

y_val = torch.LongTensor(y_val-ones_val)
y_val = torch.transpose(y_val, 0, 1)
y_val = y_val.squeeze(0)       

y_test = torch.LongTensor(y_test-ones_test)
y_test = torch.transpose(y_test, 0, 1)
y_test = y_test.squeeze(0)      

criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    X_train = X_train.cuda()
    y_train = y_train.cuda()
    X_val = X_val.cuda()
    y_val = y_val.cuda()

for i in range(POP_SIZE):
    weight, activation_function = GA_decoder_v2(pop[i,:], IND_FLOAT_LENGTH)
    bias = np.zeros([1, weight.shape[0]])
    net = AutoEncoder(weight, bias, activation_function, IND_FLOAT_LENGTH, weight.shape[0], TYPES)

    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=0)

    encoded, decoded = net.forward(X_train)

    FV[i,0] = reconstr_error(X_train, decoded, X_train.shape[0])

    for name, value in net.named_parameters():
        value.requires_grad = False
        if name == 'clf.weight'or name == 'clf.bias':
            value.requires_grad = True

    for epoch in range(EPOCH_CLF):
        optimizer.zero_grad()
        running_loss = 0.0
        predict = net.classifier(encoded)      
        loss = criterion(predict, y_train)    
        loss.backward(retain_graph=True)    
        optimizer.step() 
        running_loss += loss.item()

    FV[i,1] = CCR(net.classifier(encoded), y_train)

dorminance_matrix = 2 * np.ones([pop.shape[0],pop.shape[0]])    
for ii in range(pop.shape[0]):
    for jj in range(pop.shape[0]):
        dorminance_matrix[ii,jj] = dorminance_relation(FV[ii,:], FV[jj,:])
for ii in range(pop.shape[0]):
    individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii,:]) if x == 1]
    if len(individual_is_worse_index) == 0:
        EP.append(pop[ii,:])
        EP_FV.append(FV[ii,:])

time0_start = time.time()  
EP_and_FV_array = np.zeros([len(EP), IND_FLOAT_LENGTH+IND_BOOL_LENGTH+2])

for ii in range(EP_and_FV_array.shape[0]):
    EP_and_FV_array[ii, 0:pop.shape[1]] = EP[ii]
    EP_and_FV_array[ii, pop.shape[1]:(pop.shape[1]+2)] = EP_FV[ii]
EP_and_FV_array_unique = np.unique(EP_and_FV_array, axis=0)  

EP = []     
EP_FV = []  
for ii in range(EP_and_FV_array_unique.shape[0]):
    EP.append( EP_and_FV_array_unique[ii, 0:pop.shape[1]] )
    EP_FV.append(EP_and_FV_array_unique[ii, pop.shape[1]:(pop.shape[1]+2)])

time0_end = time.time()


Z[0, :] = FV.min(axis=0)    
parents = np.zeros([2, pop.shape[1]])

FV_child = np.zeros([2,2])      
children = np.zeros([2, pop.shape[1]])  

itera_time_start = time.time()
for itera in range(MAX_ITERA_TIMES):

    pop_permute = copy.deepcopy(pop)  
    FV_permute = np.zeros(FV.shape)

    for ii in range(POP_SIZE):
        np.random.shuffle(pop_permute[ii, 0:IND_FLOAT_LENGTH])
        np.random.shuffle(pop_permute[ii, IND_FLOAT_LENGTH:pop.shape[1]])

    pop_permute = ind_repair(pop_permute, pop_permute.shape[1] - 2, pop_permute.shape[1] - 1)

    for i in range(POP_SIZE):
        weight, activation_function = GA_decoder_v2(pop_permute[i, :], IND_FLOAT_LENGTH)
        bias = np.zeros([1, weight.shape[0]])
        net = AutoEncoder(weight, bias, activation_function, IND_FLOAT_LENGTH, weight.shape[0],
                          TYPES)  

        if torch.cuda.is_available():
            net = net.cuda()

        optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=0)

        encoded, decoded = net.forward(X_train)
        FV_permute[i, 0] = reconstr_error(X_train, decoded, X_train.shape[0])  

        for name, value in net.named_parameters():
            value.requires_grad = False
            if name == 'clf.weight' or name == 'clf.bias':
                value.requires_grad = True

        for epoch in range(EPOCH_CLF):
            optimizer.zero_grad()
            running_loss = 0.0
            predict = net.classifier(encoded)  
            loss = criterion(predict, y_train)           
            loss.backward(retain_graph=True) 
            optimizer.step() 
            running_loss += loss.item()

        FV_permute[i, 1] = CCR(net.classifier(encoded), y_train)

    for ii in range(POP_SIZE):
        relation_between_permute_and_pop = dorminance_relation(FV[ii, :], FV_permute[ii, :])
        if relation_between_permute_and_pop == 1:
            pop[ii, :] = pop_permute[ii, :]   
            FV[ii, :] = FV_permute[ii, :]

    Z[0, :] = FV.min(axis=0)  
    del pop_permute, FV_permute

    pop_previous = copy.deepcopy(pop)  
    for i in range(POP_SIZE):

        if random.random() <= DELTA:
            P = copy.deepcopy(B)
            neighbor = random.sample(range(T - 1), 2)
            parents[0, :] = pop[P[i, neighbor[0]], :]
            parents[1, :] = pop[P[i, neighbor[1]], :]
        else:
            P = copy.deepcopy(P_whole)
            neighbor = random.sample(range(POP_SIZE - 1), 2)
            parents[0, :] = pop[P[i, neighbor[0]], :]
            parents[1, :] = pop[P[i, neighbor[1]], :]

        rand_tmp = random.random() 
        if rand_tmp <= PC:
            children[:, 0:IND_FLOAT_LENGTH] = GA_SBX_crossover(parents[:,0:IND_FLOAT_LENGTH])   
            cross_point = random.randint(1, IND_BOOL_LENGTH-1)  
            children[:, IND_FLOAT_LENGTH:-1] = GA_crossover(parents[:, IND_FLOAT_LENGTH:-1], cross_point) 
        else:
            children = copy.deepcopy(parents)

        for j in range(IND_FLOAT_LENGTH):
            rand_tmp0 = random.random()  
            rand_tmp1 = random.random() 
            if rand_tmp0 <= PM:
                if random.random() <= 0.5:
                    children[0,j] += np.random.standard_cauchy(1)
                else:
                    children[0, j] += np.random.normal(0,1,1)
            if rand_tmp1 <= PM:
                if random.random() <= 0.5:
                    children[1,j] += np.random.standard_cauchy(1)
                else:
                    children[1, j] += np.random.normal(0,1,1)

        for j in range(IND_FLOAT_LENGTH, IND_FLOAT_LENGTH+IND_BOOL_LENGTH):
            rand_tmp0 = random.random()
            rand_tmp1 = random.random() 
            if rand_tmp0 <= PM:
                if children[0,j] == 1:
                    children[0, j] = 0
                else:
                    children[0, j] = 1
            if rand_tmp1 <= PM:
                if children[1,j] == 1:
                    children[1, j] = 0
                else:
                    children[1, j] = 1
        
        children = ind_repair(children, pop.shape[1]-2, pop.shape[1]-1)

        weight0, activation_func0 = GA_decoder_v2(children[0, :], IND_FLOAT_LENGTH)
        weight1, activation_func1 = GA_decoder_v2(children[1, :], IND_FLOAT_LENGTH)
        bias0 = np.zeros([1, weight0.shape[0]])
        bias1 = np.zeros([1, weight1.shape[0]])

        net0 = AutoEncoder(weight0, bias0, activation_func0, IND_FLOAT_LENGTH, weight0.shape[0],
                          TYPES) 
        net1 = AutoEncoder(weight1, bias1, activation_func1, IND_FLOAT_LENGTH, weight1.shape[0],
                           TYPES)  

        if torch.cuda.is_available():
            net0 = net0.cuda()
            net1 = net1.cuda()

        optimizer = optim.Adam(net0.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=0)

        encoded0, decoded0 = net0.forward(X_train)
        encoded1, decoded1 = net1.forward(X_train)
        FV_child[0, 0] = reconstr_error(X_train, decoded0, X_train.shape[0])  
        FV_child[1, 0] = reconstr_error(X_train, decoded1, X_train.shape[0])  


        for name, value in net0.named_parameters():
            value.requires_grad = False
            if name == 'clf.weight' or name == 'clf.bias':
                value.requires_grad = True

        for name, value in net1.named_parameters():
            value.requires_grad = False
            if name == 'clf.weight' or name == 'clf.bias':
                value.requires_grad = True

        for epoch in range(EPOCH_CLF):
            optimizer.zero_grad()
            running_loss0 = 0.0
            predict0 = net0.classifier(encoded0) 
            loss0 = criterion(predict0, y_train)  
            loss0.backward(retain_graph=True)  
            optimizer.step() 
            running_loss0 += loss0.item()

        optimizer = optim.Adam(net1.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=0)

        for epoch in range(EPOCH_CLF):
            optimizer.zero_grad()
            running_loss1 = 0.0
            predict1 = net1.classifier(encoded1)  
            loss1 = criterion(predict1, y_train)  
            loss1.backward(retain_graph=True)  
            optimizer.step()  
            running_loss1 += loss1.item()

        FV_child[0, 1] = CCR(net0.classifier(encoded0), y_train)    
        FV_child[1, 1] = CCR(net1.classifier(encoded1), y_train)   

        dorminate_flag = dorminance_relation(FV_child[0,:], FV_child[1,:])  

        if dorminate_flag == 0:
            for _ in range(2):  
                if FV_child[0, _] < Z[0, _]:
                    Z[0, _] = FV_child[0, _]

            c = 0  
            np.random.shuffle(P[i, :])      
            for j in range(P.shape[1]):
                FV_max = FV.max(axis=0)
                flag = MOEAD_ACD_better_relation(weight_vectors.values[P[i, j], :], direction_vectors[P[i, j], :],
                                                 FV_child[0, :], FV[P[i, j], :], FV_max, Z, theta[P[i, j], 0])

                if flag == 0:
                    pop[P[i, j], :] = children[0, :]
                    FV[P[i, j], :] = FV_child[0, :]
                    c += 1
                if c >= NR:
                    break

           
            if len(EP) == 0:
                for ii in range(pop.shape[0]):
                    for jj in range(pop.shape[0]):
                        dorminance_matrix[ii,jj] = dorminance_relation(FV[ii,:], FV[jj,:])
                for ii in range(pop.shape[0]):
                    individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii,:]) if x == 1]
                    if len(individual_is_worse_index) == 0:
                        EP.append(pop[ii,:])
                        EP_FV.append(FV[ii,:])

                dorminance_vectors = []  
                for EP_index in range(len(EP)):
                    dorminance_vectors.append(dorminance_relation(FV_child[0, :], EP_FV[EP_index]))
                child_is_better_index = [index for index, x in enumerate(dorminance_vectors) if
                                         x == 0]  
                child_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if
                                        x == 1]  


                for _ in range(len(child_is_better_index) - 1, -1, -1):
                    EP.pop(child_is_better_index[_])
                    EP_FV.pop(child_is_better_index[_])
                if len(child_is_worse_index) == 0:
                    EP.append(children[0, :])
                    EP_FV.append(FV_child[0, :])

            else:
                EP_in_pop = []     
                EP_in_pop_FV = []   
                for ii in range(pop.shape[0]):
                    for jj in range(pop.shape[0]):
                        dorminance_matrix[ii, jj] = dorminance_relation(FV[ii, :], FV[jj, :])
                for ii in range(pop.shape[0]):
                    individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii, :]) if x == 1]
                    if len(individual_is_worse_index) == 0:
                        EP_in_pop.append(pop[ii, :])       
                        EP_in_pop_FV.append(FV[ii, :])

                for ii in range(len(EP_in_pop)):
                    dorminance_vectors = [] 
                    for EP_index in range(len(EP)):
                        dorminance_vectors.append(dorminance_relation(EP_in_pop_FV[ii], EP_FV[EP_index]))
                    individual_is_better_index = [index for index, x in enumerate(dorminance_vectors) if
                                             x == 0]  
                    individual_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if
                                            x == 1]  
 
                    for _ in range(len(individual_is_better_index) - 1, -1, -1):
                        EP.pop(individual_is_better_index[_])
                        EP_FV.pop(individual_is_better_index[_])
                    if len(individual_is_worse_index) == 0:
                        EP.append(EP_in_pop[ii])
                        EP_FV.append(EP_in_pop_FV[ii])

                dorminance_vectors = []
                for EP_index in range(len(EP)):
                    dorminance_vectors.append(dorminance_relation(FV_child[0,:], EP_FV[EP_index]))
                child_is_better_index = [index for index, x in enumerate(dorminance_vectors) if x == 0]   
                child_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if x == 1]     

                for _ in range(len(child_is_better_index)-1,-1,-1):
                    EP.pop(child_is_better_index[_])
                    EP_FV.pop(child_is_better_index[_])

                if len(child_is_worse_index) == 0:
                    EP.append(children[0,:])
                    EP_FV.append(FV_child[0,:])

        elif dorminate_flag == 1:
            for _ in range(2):  
                if FV_child[1, _] < Z[0, _]:
                    Z[0, _] = FV_child[1, _]

            c = 0  
            np.random.shuffle(P[i, :])  
            for j in range(P.shape[1]):

                FV_max = FV.max(axis=0)
                flag = MOEAD_ACD_better_relation(weight_vectors.values[P[i, j], :], direction_vectors[P[i, j], :],
                                                 FV_child[1, :], FV[P[i, j], :], FV_max, Z, theta[P[i, j], 0])

                if flag == 0:
                    pop[P[i, j], :] = children[1, :]
                    FV[P[i, j], :] = FV_child[1, :]
                    c += 1
                if c >= NR:
                    break

            
            if len(EP) == 0:
                for ii in range(pop.shape[0]):
                    for jj in range(pop.shape[0]):
                        dorminance_matrix[ii, jj] = dorminance_relation(FV[ii, :], FV[jj, :])
                for ii in range(pop.shape[0]):
                    individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii, :]) if x == 1]
                    if len(individual_is_worse_index) == 0:
                        EP.append(pop[ii, :])
                        EP_FV.append(FV[ii, :])
						
                dorminance_vectors = [] 
                for EP_index in range(len(EP)):
                    dorminance_vectors.append(dorminance_relation(FV_child[1, :], EP_FV[EP_index]))
                child_is_better_index = [index for index, x in enumerate(dorminance_vectors) if
                                         x == 0]
                child_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if
                                        x == 1]

                for _ in range(len(child_is_better_index) - 1, -1, -1):
                    EP.pop(child_is_better_index[_])
                    EP_FV.pop(child_is_better_index[_])

                if len(child_is_worse_index) == 0:
                    EP.append(children[1, :])
                    EP_FV.append(FV_child[1, :])


            else:
                EP_in_pop = [] 
                EP_in_pop_FV = []  
                for ii in range(pop.shape[0]):
                    for jj in range(pop.shape[0]):
                        dorminance_matrix[ii, jj] = dorminance_relation(FV[ii, :], FV[jj, :])
                for ii in range(pop.shape[0]):
                    individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii, :]) if x == 1]
                    if len(individual_is_worse_index) == 0:
                        EP_in_pop.append(pop[ii, :])
                        EP_in_pop_FV.append(FV[ii, :])


                for ii in range(len(EP_in_pop)):
                    dorminance_vectors = [] 
                    for EP_index in range(len(EP)):
                        dorminance_vectors.append(dorminance_relation(EP_in_pop_FV[ii], EP_FV[EP_index]))
                    individual_is_better_index = [index for index, x in enumerate(dorminance_vectors) if
                                                  x == 0]  
                    individual_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if
                                                 x == 1]  
                   
                    for _ in range(len(individual_is_better_index) - 1, -1, -1):
                        EP.pop(individual_is_better_index[_])
                        EP_FV.pop(individual_is_better_index[_])

                    if len(individual_is_worse_index) == 0:
                        EP.append(EP_in_pop[ii])
                        EP_FV.append(EP_in_pop_FV[ii])

                dorminance_vectors = []
                for EP_index in range(len(EP)):
                    dorminance_vectors.append(dorminance_relation(FV_child[1,:], EP_FV[EP_index]))
                child_is_better_index = [index for index, x in enumerate(dorminance_vectors) if x == 0]    
                child_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if x == 1]    

                for _ in range(len(child_is_better_index)-1,-1,-1):
                    EP.pop(child_is_better_index[_])
                    EP_FV.pop(child_is_better_index[_])
                if len(child_is_worse_index) == 0:
                    EP.append(children[1,:])
                    EP_FV.append(FV_child[1,:])

        elif dorminate_flag == 2:
            if FV_child[0,0] <= FV_child[1,0]:
                z0 = FV_child[0,0]
            else:
                z0 = FV_child[1,0]

            if FV_child[0,1] <= FV_child[1,1]:
                z1 = FV_child[0,1]
            else:
                z1 = FV_child[1,1]

            if z0 < Z[0, 0]:
                Z[0, 0] = z0
            if z1 < Z[0, 1]:
                Z[0, 1] = z1

            c = 0  
            np.random.shuffle(P[i, :]) 
            for j in range(P.shape[1]):

                FV_max = FV.max(axis=0)

                flag0 = MOEAD_ACD_better_relation(weight_vectors.values[P[i, j], :], direction_vectors[P[i, j], :],
                                                 FV_child[0, :], FV[P[i, j], :], FV_max, Z, theta[P[i, j], 0])
                flag1 = MOEAD_ACD_better_relation(weight_vectors.values[P[i, j], :], direction_vectors[P[i, j], :],
                                                  FV_child[1, :], FV[P[i, j], :], FV_max, Z, theta[P[i, j], 0])

                if flag0 == 0:
                    pop[P[i, j], :] = children[0, :]
                    FV[P[i, j], :] = FV_child[0, :]
                    c += 1
                if c >= NR:
                    break
                if flag0 == 1:
                    pop[P[i, j], :] = children[1, :]
                    FV[P[i, j], :] = FV_child[1, :]
                    c += 1
                if c >= NR:
                    break

          
            if len(EP) == 0:
                for ii in range(pop.shape[0]):
                    for jj in range(pop.shape[0]):
                        dorminance_matrix[ii, jj] = dorminance_relation(FV[ii, :], FV[jj, :])
                for ii in range(pop.shape[0]):
                    individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii, :]) if x == 1]
                    if len(individual_is_worse_index) == 0:
                        EP.append(pop[ii, :])
                        EP_FV.append(FV[ii, :])

                dorminance_vectors0 = []  
                dorminance_vectors1 = []  
                for EP_index in range(len(EP)):
                    dorminance_vectors0.append(dorminance_relation(FV_child[0, :], EP_FV[EP_index]))
                    dorminance_vectors1.append(dorminance_relation(FV_child[1, :], EP_FV[EP_index]))
                child0_is_better_index = [index for index, x in enumerate(dorminance_vectors0) if
                                          x == 0]  
                child1_is_better_index = [index for index, x in enumerate(dorminance_vectors1) if
                                          x == 0] 
                child0_is_worse_index = [index for index, x in enumerate(dorminance_vectors0) if
                                         x == 1]  
                child1_is_worse_index = [index for index, x in enumerate(dorminance_vectors1) if
                                         x == 1]  

                for _ in range(len(child0_is_better_index) - 1, -1, -1):
                    EP[child0_is_better_index[_]] = 0
                for _ in range(len(child1_is_better_index) - 1, -1, -1):
                    EP[child1_is_better_index[_]] = 0
                for _ in range(len(EP) - 1, -1, -1):
                    if type(EP[_]) == int:
                        if EP[_] == 0:
                            EP.pop(_)
                            EP_FV.pop(_)
                if len(child0_is_worse_index) == 0:
                    EP.append(children[0, :])
                    EP_FV.append(FV_child[0, :])
                if len(child1_is_worse_index) == 0:
                    EP.append(children[1, :])
                    EP_FV.append(FV_child[1, :])

            else:
                EP_in_pop = []  
                EP_in_pop_FV = []  
                for ii in range(pop.shape[0]):
                    for jj in range(pop.shape[0]):
                        dorminance_matrix[ii, jj] = dorminance_relation(FV[ii, :], FV[jj, :])
                for ii in range(pop.shape[0]):
                    individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii, :]) if x == 1]
                    if len(individual_is_worse_index) == 0:
                        EP_in_pop.append(pop[ii, :])
                        EP_in_pop_FV.append(FV[ii, :])

                for ii in range(len(EP_in_pop)):
                    dorminance_vectors = []  
                    for EP_index in range(len(EP)):
                        dorminance_vectors.append(dorminance_relation(EP_in_pop_FV[ii], EP_FV[EP_index]))
                    individual_is_better_index = [index for index, x in enumerate(dorminance_vectors) if
                                                  x == 0]  
                    individual_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if
                                                 x == 1]  

                    for _ in range(len(individual_is_better_index) - 1, -1, -1):
                        EP.pop(individual_is_better_index[_])
                        EP_FV.pop(individual_is_better_index[_])

                    if len(individual_is_worse_index) == 0:
                        EP.append(EP_in_pop[ii])
                        EP_FV.append(EP_in_pop_FV[ii])

                dorminance_vectors0 = []  
                dorminance_vectors1 = []  
                for EP_index in range(len(EP)):
                    dorminance_vectors0.append(dorminance_relation(FV_child[0, :], EP_FV[EP_index]))
                    dorminance_vectors1.append(dorminance_relation(FV_child[1, :], EP_FV[EP_index]))
                child0_is_better_index = [index for index, x in enumerate(dorminance_vectors0) if
                                         x == 0] 
                child1_is_better_index = [index for index, x in enumerate(dorminance_vectors1) if
                                          x == 0]  
                child0_is_worse_index = [index for index, x in enumerate(dorminance_vectors0) if
                                        x == 1]  
                child1_is_worse_index = [index for index, x in enumerate(dorminance_vectors1) if
                                        x == 1]  

                for _ in range(len(child0_is_better_index) - 1, -1, -1):
                    EP[child0_is_better_index[_]] = 0
                for _ in range(len(child1_is_better_index) - 1, -1, -1):
                    EP[child1_is_better_index[_]] = 0
                for _ in range(len(EP)-1, -1, -1):
                    if type(EP[_]) == int:
                        if EP[_] == 0:
                            EP.pop(_)
                            EP_FV.pop(_)
                if len(child0_is_worse_index) == 0:
                    EP.append(children[0, :])
                    EP_FV.append(FV_child[0, :])
                if len(child1_is_worse_index) == 0:
                    EP.append(children[1, :])
                    EP_FV.append(FV_child[1, :])

        EP_and_FV_array = np.zeros([len(EP), IND_FLOAT_LENGTH + IND_BOOL_LENGTH + 2])

        for ii in range(EP_and_FV_array.shape[0]):
            EP_and_FV_array[ii, 0:pop.shape[1]] = EP[ii]
            EP_and_FV_array[ii, pop.shape[1]:(pop.shape[1] + 2)] = EP_FV[ii]
        EP_and_FV_array_unique = np.unique(EP_and_FV_array, axis=0) 

        EP = [] 
        EP_FV = []  
        for ii in range(EP_and_FV_array_unique.shape[0]):
            EP.append(EP_and_FV_array_unique[ii, 0:pop.shape[1]])
            EP_FV.append(EP_and_FV_array_unique[ii, pop.shape[1]:(pop.shape[1] + 2)])

    theta = adaptive_adjust_theta(theta, theta_min, theta_max, FV, direction_vectors, Z)

itera_time_end = time.time()

if len(EP_FV) <= 2:
    knee_point_individual = find_closest_point(EP, EP_FV, Z)
else:
    knee_point_individual = find_knee_point(EP, EP_FV)

weight, activation_function = GA_decoder_v2(knee_point_individual, IND_FLOAT_LENGTH)      
bias = np.zeros([1, weight.shape[0]])
net = AutoEncoder(weight, bias, activation_function, IND_FLOAT_LENGTH, weight.shape[0], TYPES)    

activation_func_list.append(activation_function)  
hidden_dim_list.append(IND_FLOAT_LENGTH)      
hidden_dim_list.append(weight.shape[0])     

if torch.cuda.is_available():
    net = net.cuda()
    print('======= train the net by GPU =======')

optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=0)
criterion = nn.CrossEntropyLoss()    
val_best_loss_per_layer = []    
val_best_loss = 100 
fail_time = 0   

for epoch in range(MAX_EPOCH):
    optimizer.zero_grad()
    running_loss = 0.0
    encoded_train, decoded_train = net.forward(X_train)  
    predict_train = net.classifier(encoded_train)  
    loss = criterion(predict_train, y_train)     
    loss.backward(retain_graph=True)  
    optimizer.step() 
    running_loss += loss.item()
    if epoch % VALID_FREQ == 0 and epoch % (4 * VALID_FREQ) == 0:
        print('the first hidden layer, train set %d epoch, loss: %.6f' % (epoch, running_loss))

    if epoch % VALID_FREQ == 0:
        encoded_val, decoded_val = net.forward(X_val)  
        predict_val = net.classifier(encoded_val)  
        loss_val = criterion(predict_val, y_val)

        if epoch % (4 * VALID_FREQ) == 0:
            print('the first hidden layer, validataion set %d epoch, loss: %.6f' % (epoch, loss_val.item()))

        if loss_val.item() >= val_best_loss:
            fail_time += 1
        else:
            fail_time = 0
            val_best_loss = loss_val.item()

        if fail_time >= VALID_PATIENCE:
            print('======= early stop =======')
            val_best_loss_per_layer.append(val_best_loss)  
            break
    if epoch == (MAX_EPOCH - 1):
        val_best_loss_per_layer.append(val_best_loss)
        print('val_best_loss_per_layer:\n', val_best_loss_per_layer)

X_train_encoded, X_train_decoded = net.forward(X_train)
X_val_encoded, X_val_decoded = net.forward(X_val)
del X_train_decoded, X_val_decoded


for k, v in net.state_dict().items():

    if k == 'encoder.0.weight':
        weight_list.append(v.cpu()) 
    elif k == 'encoder.0.bias':
        bias_list.append(v.cpu()) 

for layer_num in range(2, MAX_HIDDEN_LAYER_NUM+1):

    for name, value in net.named_parameters():
        if layer_num == 2:
            if name == 'encoder.0.weight':
                start_dim = value.shape[0]
        else:
            if name == 'encoder.encoder_layer_'+str(layer_num-2)+'.weight':
                start_dim = value.shape[0]

    ind_float = np.random.random((weight_vectors.shape[0], start_dim))
    print('X_train_encode.shape: ', X_train_encoded.shape)
    print('start_dim: ', start_dim)
    ind_bool = np.random.randint(0, 2, (weight_vectors.shape[0], start_dim+1))
    pop = np.hstack((ind_float, ind_bool))
    print('pop.shape:\n', pop.shape)

    pop = ind_repair(pop, pop.shape[1]-2, pop.shape[1]-1)

    FV = np.zeros((POP_SIZE, OBJ_NUM))
    FV_normalize = np.zeros((POP_SIZE, OBJ_NUM)) 
    Z = np.zeros((1, OBJ_NUM))

    for i in range(POP_SIZE):
        weight, activation_function = GA_decoder_v2(pop[i,:], start_dim)
        bias = np.zeros([1, weight.shape[0]])
        net = AutoEncoder(weight, bias, activation_function, start_dim, weight.shape[0], TYPES)

        if torch.cuda.is_available():
            net = net.cuda()

        # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=0)

        encoded, decoded = net.forward(X_train_encoded) 
        FV[i,0] = reconstr_error(X_train_encoded, decoded, X_train_encoded.shape[0]) 

        for name, value in net.named_parameters():
            value.requires_grad = False
            if name == 'clf.weight'or name == 'clf.bias':
                value.requires_grad = True

        for epoch in range(EPOCH_CLF):
            optimizer.zero_grad()
            running_loss = 0.0
            predict = net.classifier(encoded)     
            loss = criterion(predict, y_train)    
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()

        FV[i,1] = CCR(net.classifier(encoded), y_train)

    EP = [] 
    EP_FV = []

    dorminance_matrix = 2 * np.ones([pop.shape[0],pop.shape[0]])
    for ii in range(pop.shape[0]):
        for jj in range(pop.shape[0]):
            dorminance_matrix[ii,jj] = dorminance_relation(FV[ii,:], FV[jj,:])
    for ii in range(pop.shape[0]):
        individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii,:]) if x == 1]
        if len(individual_is_worse_index) == 0:
            EP.append(pop[ii,:])
            EP_FV.append(FV[ii,:])

    time0_start = time.time()  
    EP_and_FV_array = np.zeros([len(EP), pop.shape[1]+2])

    for ii in range(EP_and_FV_array.shape[0]):
        EP_and_FV_array[ii, 0:pop.shape[1]] = EP[ii]
        EP_and_FV_array[ii, pop.shape[1]:(pop.shape[1]+2)] = EP_FV[ii]
    EP_and_FV_array_unique = np.unique(EP_and_FV_array, axis=0) 

    EP = []  
    EP_FV = [] 
    for ii in range(EP_and_FV_array_unique.shape[0]):
        EP.append( EP_and_FV_array_unique[ii, 0:pop.shape[1]] )
        EP_FV.append(EP_and_FV_array_unique[ii, pop.shape[1]:(pop.shape[1]+2)])

    time0_end = time.time()

    Z[0, :] = FV.min(axis=0)  
    parents = np.zeros([2, pop.shape[1]])  

    FV_child = np.zeros([2,2])  
    children = np.zeros([2, pop.shape[1]]) 

    itera_time_start = time.time()
    for itera in range(MAX_ITERA_TIMES):
        pop_permute = copy.deepcopy(pop)
        FV_permute = np.zeros(FV.shape)


        for ii in range(POP_SIZE):
            np.random.shuffle(pop_permute[ii, 0:start_dim])
            np.random.shuffle(pop_permute[ii, start_dim:pop.shape[1]])

        pop_permute = ind_repair(pop_permute, pop_permute.shape[1] - 2, pop_permute.shape[1] - 1)

        for i in range(POP_SIZE):
            weight, activation_function = GA_decoder_v2(pop_permute[i, :], start_dim)
            bias = np.zeros([1, weight.shape[0]])
            net = AutoEncoder(weight, bias, activation_function, start_dim, weight.shape[0],
                              TYPES)  

            if torch.cuda.is_available():
                net = net.cuda()

            # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
            optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=0)
            encoded, decoded = net.forward(X_train_encoded)
            FV_permute[i, 0] = reconstr_error(X_train_encoded, decoded, X_train.shape[0])  
			
            for name, value in net.named_parameters():
                value.requires_grad = False
                if name == 'clf.weight' or name == 'clf.bias':
                    value.requires_grad = True

            for epoch in range(EPOCH_CLF):
                optimizer.zero_grad()
                running_loss = 0.0
                predict = net.classifier(encoded)  
                loss = criterion(predict, y_train)  
                loss.backward(retain_graph=True) 
                optimizer.step()  
                running_loss += loss.item()

            FV_permute[i, 1] = CCR(net.classifier(encoded), y_train)

        for ii in range(POP_SIZE):
            relation_between_permute_and_pop = dorminance_relation(FV[ii, :], FV_permute[ii, :])
            if relation_between_permute_and_pop == 1:
                pop[ii, :] = pop_permute[ii, :]
                FV[ii, :] = FV_permute[ii, :]

        Z[0, :] = FV.min(axis=0) 
        del pop_permute, FV_permute

        pop_previous = copy.deepcopy(pop) 
        for i in range(POP_SIZE):

            if random.random() <= DELTA:
                P = copy.deepcopy(B)
                neighbor = random.sample(range(T - 1), 2)
                parents[0, :] = pop[P[i, neighbor[0]], :]
                parents[1, :] = pop[P[i, neighbor[1]], :]
            else:
                P = copy.deepcopy(P_whole)
                neighbor = random.sample(range(POP_SIZE - 1), 2)
                parents[0, :] = pop[P[i, neighbor[0]], :]
                parents[1, :] = pop[P[i, neighbor[1]], :]

            rand_tmp = random.random() 
            if rand_tmp <= PC:
                children[:, 0:start_dim] = GA_SBX_crossover(parents[:,0:start_dim])   
                cross_point = random.randint(1, start_dim-1) 
                children[:, start_dim:-1] = GA_crossover(parents[:, start_dim:-1], cross_point) 
            else:
                children = copy.deepcopy(parents)
				
            for j in range(start_dim):
                rand_tmp0 = random.random()  
                rand_tmp1 = random.random()  
                if rand_tmp0 <= PM:
                    if random.random() <= 0.5:
                        children[0,j] += np.random.standard_cauchy(1)
                    else:
                        children[0, j] += np.random.normal(0,1,1)
                if rand_tmp1 <= PM:
                    if random.random() <= 0.5:
                        children[1,j] += np.random.standard_cauchy(1)
                    else:
                        children[1, j] += np.random.normal(0,1,1)

            for j in range(start_dim, pop.shape[1]):
                rand_tmp0 = random.random()  
                rand_tmp1 = random.random()  
                if rand_tmp0 <= PM:
                    if children[0,j] == 1:
                        children[0, j] = 0
                    else:
                        children[0, j] = 1
                if rand_tmp1 <= PM:
                    if children[1,j] == 1:
                        children[1, j] = 0
                    else:
                        children[1, j] = 1

            children = ind_repair(children, pop.shape[1]-2, pop.shape[1]-1)

            weight0, activation_func0 = GA_decoder_v2(children[0, :], start_dim)
            weight1, activation_func1 = GA_decoder_v2(children[1, :], start_dim)
            bias0 = np.zeros([1, weight0.shape[0]])
            bias1 = np.zeros([1, weight1.shape[0]])

            net0 = AutoEncoder(weight0, bias0, activation_func0, start_dim, weight0.shape[0],
                              TYPES) 
            net1 = AutoEncoder(weight1, bias1, activation_func1, start_dim, weight1.shape[0],
                               TYPES) 

            if torch.cuda.is_available():
                net0 = net0.cuda()
                net1 = net1.cuda()

            # optimizer = optim.SGD(net0.parameters(), lr=LR, momentum=0.9)
            optimizer = optim.Adam(net0.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=0)

            encoded0, decoded0 = net0.forward(X_train_encoded)
            encoded1, decoded1 = net1.forward(X_train_encoded)
            FV_child[0, 0] = reconstr_error(X_train_encoded, decoded0, X_train_encoded.shape[0])    
            FV_child[1, 0] = reconstr_error(X_train_encoded, decoded1, X_train_encoded.shape[0])    

            for name, value in net0.named_parameters():
                value.requires_grad = False
                if name == 'clf.weight' or name == 'clf.bias':
                    value.requires_grad = True

            for name, value in net1.named_parameters():
                value.requires_grad = False
                if name == 'clf.weight' or name == 'clf.bias':
                    value.requires_grad = True

            for epoch in range(EPOCH_CLF):
                optimizer.zero_grad()
                running_loss0 = 0.0
                predict0 = net0.classifier(encoded0)  
                loss0 = criterion(predict0, y_train) 
                loss0.backward(retain_graph=True)  
                optimizer.step()  
                running_loss0 += loss0.item()

            # optimizer = optim.SGD(net1.parameters(), lr=LR, momentum=0.9)
            optimizer = optim.Adam(net1.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=0)

            for epoch in range(EPOCH_CLF):
                optimizer.zero_grad()
                running_loss1 = 0.0
                predict1 = net1.classifier(encoded1)  
                loss1 = criterion(predict1, y_train)
                loss1.backward(retain_graph=True)  
                optimizer.step()  
                running_loss1 += loss1.item()

            FV_child[0, 1] = CCR(net0.classifier(encoded0), y_train)    
            FV_child[1, 1] = CCR(net1.classifier(encoded1), y_train)   

            dorminate_flag = dorminance_relation(FV_child[0,:], FV_child[1,:])  
            if dorminate_flag == 0:
                for _ in range(2): 
                    if FV_child[0, _] < Z[0, _]:
                        Z[0, _] = FV_child[0, _]
                c = 0
                np.random.shuffle(P[i, :]) 
                for j in range(P.shape[1]):
                    FV_max = FV.max(axis=0)
                    flag = MOEAD_ACD_better_relation(weight_vectors.values[P[i, j], :],
                                                     direction_vectors[P[i, j], :],
                                                     FV_child[0, :], FV[P[i, j], :], FV_max, Z, theta[P[i, j], 0])

                    if flag == 0:
                        pop[P[i, j], :] = children[0, :]
                        FV[P[i, j], :] = FV_child[0, :]
                        c += 1
                    if c >= NR:
                        break

                if len(EP) == 0:
                    for ii in range(pop.shape[0]):
                        for jj in range(pop.shape[0]):
                            dorminance_matrix[ii,jj] = dorminance_relation(FV[ii,:], FV[jj,:])
                    for ii in range(pop.shape[0]):
                        individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii,:]) if x == 1]
                        if len(individual_is_worse_index) == 0:
                            EP.append(pop[ii,:])
                            EP_FV.append(FV[ii,:])

                    dorminance_vectors = []  
                    for EP_index in range(len(EP)):
                        dorminance_vectors.append(dorminance_relation(FV_child[0, :], EP_FV[EP_index]))
                    child_is_better_index = [index for index, x in enumerate(dorminance_vectors) if
                                             x == 0] 
                    child_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if
                                            x == 1]  

                    for _ in range(len(child_is_better_index) - 1, -1, -1):
                        EP.pop(child_is_better_index[_])
                        EP_FV.pop(child_is_better_index[_])
                    if len(child_is_worse_index) == 0:
                        EP.append(children[0, :])
                        EP_FV.append(FV_child[0, :])

                else:
                    EP_in_pop = []  
                    EP_in_pop_FV = [] 
                    for ii in range(pop.shape[0]):
                        for jj in range(pop.shape[0]):
                            dorminance_matrix[ii, jj] = dorminance_relation(FV[ii, :], FV[jj, :])
                    for ii in range(pop.shape[0]):
                        individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii, :]) if x == 1]
                        if len(individual_is_worse_index) == 0:
                            EP_in_pop.append(pop[ii, :]) 
                            EP_in_pop_FV.append(FV[ii, :])

                    for ii in range(len(EP_in_pop)):
                        dorminance_vectors = []
                        for EP_index in range(len(EP)):
                            dorminance_vectors.append(dorminance_relation(EP_in_pop_FV[ii], EP_FV[EP_index]))
                        individual_is_better_index = [index for index, x in enumerate(dorminance_vectors) if
                                                 x == 0] 
                        individual_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if
                                                x == 1] 
                        for _ in range(len(individual_is_better_index) - 1, -1, -1):
                            EP.pop(individual_is_better_index[_])
                            EP_FV.pop(individual_is_better_index[_])
                        if len(individual_is_worse_index) == 0:
                            EP.append(EP_in_pop[ii])
                            EP_FV.append(EP_in_pop_FV[ii])

                    dorminance_vectors = [] 
                    for EP_index in range(len(EP)):
                        dorminance_vectors.append(dorminance_relation(FV_child[0,:], EP_FV[EP_index]))
                    child_is_better_index = [index for index, x in enumerate(dorminance_vectors) if x == 0]    
                    child_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if x == 1]   
					
                    for _ in range(len(child_is_better_index)-1,-1,-1):
                        EP.pop(child_is_better_index[_])
                        EP_FV.pop(child_is_better_index[_])
                    if len(child_is_worse_index) == 0:
                        EP.append(children[0,:])
                        EP_FV.append(FV_child[0,:])

            elif dorminate_flag == 1:
                for _ in range(2):  
                    if FV_child[1, _] < Z[0, _]:
                        Z[0, _] = FV_child[1, _]
                c = 0
                np.random.shuffle(P[i, :])  
                for j in range(P.shape[1]):
                    FV_max = FV.max(axis=0)
                    flag = MOEAD_ACD_better_relation(weight_vectors.values[P[i, j], :],
                                                     direction_vectors[P[i, j], :],
                                                     FV_child[1, :], FV[P[i, j], :], FV_max, Z, theta[P[i, j], 0])

                    if flag == 0:
                        pop[P[i, j], :] = children[1, :]
                        FV[P[i, j], :] = FV_child[1, :]
                        c += 1
                    if c >= NR:
                        break
           
                if len(EP) == 0:
                    for ii in range(pop.shape[0]):
                        for jj in range(pop.shape[0]):
                            dorminance_matrix[ii, jj] = dorminance_relation(FV[ii, :], FV[jj, :])
                    for ii in range(pop.shape[0]):
                        individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii, :]) if x == 1]
                        if len(individual_is_worse_index) == 0:
                            EP.append(pop[ii, :])
                            EP_FV.append(FV[ii, :])

                    dorminance_vectors = [] 
                    for EP_index in range(len(EP)):
                        dorminance_vectors.append(dorminance_relation(FV_child[1, :], EP_FV[EP_index]))
                    child_is_better_index = [index for index, x in enumerate(dorminance_vectors) if
                                             x == 0]  
                    child_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if
                                            x == 1]  

                    for _ in range(len(child_is_better_index) - 1, -1, -1):
                        EP.pop(child_is_better_index[_])
                        EP_FV.pop(child_is_better_index[_])

                    if len(child_is_worse_index) == 0:
                        EP.append(children[1, :])
                        EP_FV.append(FV_child[1, :])

                else:
                    EP_in_pop = []
                    EP_in_pop_FV = []  
                    for ii in range(pop.shape[0]):
                        for jj in range(pop.shape[0]):
                            dorminance_matrix[ii, jj] = dorminance_relation(FV[ii, :], FV[jj, :])
                    for ii in range(pop.shape[0]):
                        individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii, :]) if x == 1]
                        if len(individual_is_worse_index) == 0:
                            EP_in_pop.append(pop[ii, :])
                            EP_in_pop_FV.append(FV[ii, :])

                    for ii in range(len(EP_in_pop)):
                        dorminance_vectors = []  
                        for EP_index in range(len(EP)):
                            dorminance_vectors.append(dorminance_relation(EP_in_pop_FV[ii], EP_FV[EP_index]))
                        individual_is_better_index = [index for index, x in enumerate(dorminance_vectors) if
                                                      x == 0] 
                        individual_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if
                                                     x == 1]  
                        for _ in range(len(individual_is_better_index) - 1, -1, -1):
                            EP.pop(individual_is_better_index[_])
                            EP_FV.pop(individual_is_better_index[_])
                        if len(individual_is_worse_index) == 0:
                            EP.append(EP_in_pop[ii])
                            EP_FV.append(EP_in_pop_FV[ii])

                    dorminance_vectors = []
                    for EP_index in range(len(EP)):
                        dorminance_vectors.append(dorminance_relation(FV_child[1,:], EP_FV[EP_index]))
                    child_is_better_index = [index for index, x in enumerate(dorminance_vectors) if x == 0]    
                    child_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if x == 1]    

                    for _ in range(len(child_is_better_index)-1,-1,-1):
                        EP.pop(child_is_better_index[_])
                        EP_FV.pop(child_is_better_index[_])
                    if len(child_is_worse_index) == 0:
                        EP.append(children[1,:])
                        EP_FV.append(FV_child[1,:])

            elif dorminate_flag == 2:
                if FV_child[0,0] <= FV_child[1,0]:
                    z0 = FV_child[0,0]
                else:
                    z0 = FV_child[1,0]

                if FV_child[0,1] <= FV_child[1,1]:
                    z1 = FV_child[0,1]
                else:
                    z1 = FV_child[1,1]

                if z0 < Z[0, 0]:
                    Z[0, 0] = z0
                if z1 < Z[0, 1]:
                    Z[0, 1] = z1

                c = 0
                np.random.shuffle(P[i, :]) 
                for j in range(P.shape[1]):
                    FV_max = FV.max(axis=0)
                    flag0 = MOEAD_ACD_better_relation(weight_vectors.values[P[i, j], :],
                                                      direction_vectors[P[i, j], :],
                                                      FV_child[0, :], FV[P[i, j], :], FV_max, Z, theta[P[i, j], 0])
                    flag1 = MOEAD_ACD_better_relation(weight_vectors.values[P[i, j], :],
                                                      direction_vectors[P[i, j], :],
                                                      FV_child[1, :], FV[P[i, j], :], FV_max, Z, theta[P[i, j], 0])

                    if flag0 == 0:
                        pop[P[i, j], :] = children[0, :]
                        FV[P[i, j], :] = FV_child[0, :]
                        c += 1
                    if c >= NR:
                        break
                    if flag0 == 1:
                        pop[P[i, j], :] = children[1, :]
                        FV[P[i, j], :] = FV_child[1, :]
                        c += 1
                    if c >= NR:
                        break

                if len(EP) == 0:
                    for ii in range(pop.shape[0]):
                        for jj in range(pop.shape[0]):
                            dorminance_matrix[ii, jj] = dorminance_relation(FV[ii, :], FV[jj, :])
                    for ii in range(pop.shape[0]):
                        individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii, :]) if x == 1]
                        if len(individual_is_worse_index) == 0:
                            EP.append(pop[ii, :])
                            EP_FV.append(FV[ii, :])

                    dorminance_vectors0 = []  
                    dorminance_vectors1 = [] 
                    for EP_index in range(len(EP)):
                        dorminance_vectors0.append(dorminance_relation(FV_child[0, :], EP_FV[EP_index]))
                        dorminance_vectors1.append(dorminance_relation(FV_child[1, :], EP_FV[EP_index]))
                    child0_is_better_index = [index for index, x in enumerate(dorminance_vectors0) if
                                              x == 0]  
                    child1_is_better_index = [index for index, x in enumerate(dorminance_vectors1) if
                                              x == 0] 
                    child0_is_worse_index = [index for index, x in enumerate(dorminance_vectors0) if
                                             x == 1]  
                    child1_is_worse_index = [index for index, x in enumerate(dorminance_vectors1) if
                                             x == 1]  
                    for _ in range(len(child0_is_better_index) - 1, -1, -1):
                        EP[child0_is_better_index[_]] = 0
                    for _ in range(len(child1_is_better_index) - 1, -1, -1):
                        EP[child1_is_better_index[_]] = 0
                    for _ in range(len(EP) - 1, -1, -1):

                        if type(EP[_]) == int:
                            if EP[_] == 0:
                                EP.pop(_)
                                EP_FV.pop(_)
                    if len(child0_is_worse_index) == 0:
                        EP.append(children[0, :])
                        EP_FV.append(FV_child[0, :])
                    if len(child1_is_worse_index) == 0:
                        EP.append(children[1, :])
                        EP_FV.append(FV_child[1, :])

                else:
                    EP_in_pop = [] 
                    EP_in_pop_FV = []  
                    for ii in range(pop.shape[0]):
                        for jj in range(pop.shape[0]):
                            dorminance_matrix[ii, jj] = dorminance_relation(FV[ii, :], FV[jj, :])
                    for ii in range(pop.shape[0]):
                        individual_is_worse_index = [index for index, x in enumerate(dorminance_matrix[ii, :]) if x == 1]
                        if len(individual_is_worse_index) == 0:
                            EP_in_pop.append(pop[ii, :])
                            EP_in_pop_FV.append(FV[ii, :])

                    for ii in range(len(EP_in_pop)):
                        dorminance_vectors = []  
                        for EP_index in range(len(EP)):
                            dorminance_vectors.append(dorminance_relation(EP_in_pop_FV[ii], EP_FV[EP_index]))
                        individual_is_better_index = [index for index, x in enumerate(dorminance_vectors) if
                                                      x == 0]  
                        individual_is_worse_index = [index for index, x in enumerate(dorminance_vectors) if
                                                     x == 1]  
                        for _ in range(len(individual_is_better_index) - 1, -1, -1):
                            EP.pop(individual_is_better_index[_])
                            EP_FV.pop(individual_is_better_index[_])
                        if len(individual_is_worse_index) == 0:
                            EP.append(EP_in_pop[ii])
                            EP_FV.append(EP_in_pop_FV[ii])

                    dorminance_vectors0 = [] 
                    dorminance_vectors1 = []
                    for EP_index in range(len(EP)):
                        dorminance_vectors0.append(dorminance_relation(FV_child[0, :], EP_FV[EP_index]))
                        dorminance_vectors1.append(dorminance_relation(FV_child[1, :], EP_FV[EP_index]))
                    child0_is_better_index = [index for index, x in enumerate(dorminance_vectors0) if
                                             x == 0]  
                    child1_is_better_index = [index for index, x in enumerate(dorminance_vectors1) if
                                              x == 0]  
                    child0_is_worse_index = [index for index, x in enumerate(dorminance_vectors0) if
                                            x == 1]  
                    child1_is_worse_index = [index for index, x in enumerate(dorminance_vectors1) if
                                            x == 1]  

                    for _ in range(len(child0_is_better_index) - 1, -1, -1):
                        EP[child0_is_better_index[_]] = 0
                    for _ in range(len(child1_is_better_index) - 1, -1, -1):
                        EP[child1_is_better_index[_]] = 0
                    for _ in range(len(EP)-1, -1, -1):
                        if type(EP[_]) == int:
                            if EP[_] == 0:
                                EP.pop(_)
                                EP_FV.pop(_)
                    if len(child0_is_worse_index) == 0:
                        EP.append(children[0, :])
                        EP_FV.append(FV_child[0, :])
                    if len(child1_is_worse_index) == 0:
                        EP.append(children[1, :])
                        EP_FV.append(FV_child[1, :])

           
            EP_and_FV_array = np.zeros([len(EP), pop.shape[1]+2])

            for ii in range(EP_and_FV_array.shape[0]):
                EP_and_FV_array[ii, 0:pop.shape[1]] = EP[ii]
                EP_and_FV_array[ii, pop.shape[1]:(pop.shape[1] + 2)] = EP_FV[ii]
            EP_and_FV_array_unique = np.unique(EP_and_FV_array, axis=0)  

            EP = [] 
            EP_FV = []  
            for ii in range(EP_and_FV_array_unique.shape[0]):
                EP.append(EP_and_FV_array_unique[ii, 0:pop.shape[1]])
                EP_FV.append(EP_and_FV_array_unique[ii, pop.shape[1]:(pop.shape[1] + 2)])

        theta = adaptive_adjust_theta(theta, theta_min, theta_max, FV, direction_vectors, Z)

    itera_time_end = time.time()

    if len(EP_FV) <= 2:
        knee_point_individual = find_closest_point(EP, EP_FV, Z)
    else:
        knee_point_individual = find_knee_point(EP, EP_FV)

    weight, activation_function = GA_decoder_v2(knee_point_individual, start_dim)
    bias = np.zeros([1, weight.shape[0]])

    print('bias.shape: ', bias.shape)
    weight_list.append(weight)
    bias_list.append(bias)
    activation_func_list.append(activation_function)
    hidden_dim_list.append(weight.shape[0])
    print('len(weight_list):\n', len(weight_list))
    print('len(hidden_dim_list):\n', len(hidden_dim_list))

    net = AutoEncoder_total(weight_list, bias_list, activation_func_list, hidden_dim_list, TYPES)

    if torch.cuda.is_available():
        net = net.cuda()
        print('======= train the net by GPU =======')

    optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.999), lr=LR, weight_decay=0)
    criterion = nn.CrossEntropyLoss() 
    val_best_loss = 100 
    fail_time = 0

    for epoch in range(MAX_EPOCH):
        optimizer.zero_grad()
        running_loss = 0.0
        encoded_train = net.forward(X_train)
        predict_train = net.classifier(encoded_train) 
        loss = criterion(predict_train, y_train) 
        loss.backward(retain_graph=True) 
        optimizer.step()  
        running_loss += loss.item()
        if epoch % VALID_FREQ == 0 and epoch % (4 * VALID_FREQ) == 0:
            print('the %d hidden layer, train set %d epoch, loss: %.6f' % (layer_num, epoch, running_loss))

        if epoch % VALID_FREQ == 0:
            encoded_val = net.forward(X_val)  
            predict_val = net.classifier(encoded_val)  
            loss_val = criterion(predict_val, y_val)

            if epoch % (4 * VALID_FREQ) == 0:
                print('the %d hidden layer, validation set %d epoch, loss: %.6f' % (layer_num, epoch, loss_val.item()))

            if loss_val.item() >= val_best_loss:
                fail_time += 1
            else:
                fail_time = 0
                val_best_loss = loss_val.item()

            if fail_time >= VALID_PATIENCE:
                val_best_loss_per_layer.append(val_best_loss) 
                print('val_best_loss_per_layer:\n', val_best_loss_per_layer)

                break
        if epoch == (MAX_EPOCH-1):
            val_best_loss_per_layer.append(val_best_loss)  
            print('val_best_loss_per_layer:\n', val_best_loss_per_layer)

    print('val_best_loss_per_layer:\n', val_best_loss_per_layer)
    print('len(val_best_loss_per_layer): ', len(val_best_loss_per_layer))
    if val_best_loss_per_layer[len(val_best_loss_per_layer)-1] > val_best_loss_per_layer[len(val_best_loss_per_layer)-2]:
        weight_list.pop()
        bias_list.pop()
        activation_func_list.pop()
        hidden_dim_list.pop()
        break

    X_train_encoded = net(X_train)
    X_val_encoded = net(X_val)

    weight_list = []
    bias_list = []

    assign_No = 0 
    for k, v in net.state_dict().items():
        if assign_No % 2 == 0:
            if len(weight_list) < layer_num:
                weight_list.append(v.cpu())
                assign_No += 1
        else:
            if len(bias_list) < layer_num:
                bias_list.append(v.cpu())
                assign_No += 1
del net

net_final = AutoEncoder_total(weight_list, bias_list, activation_func_list, hidden_dim_list, TYPES)
print('net_final:\n', net_final)
print('hidden_dim_list: \n', hidden_dim_list)
print('activation_func_list: \n', activation_func_list)

if torch.cuda.is_available():
    net_final = net_final.cuda()
    print('======= train the net by GPU =======')

val_best_loss = 100 
fail_time = 0  
optimizer = optim.Adam(net_final.parameters(), betas=(0.9, 0.999), lr=LR_FINAL, weight_decay=0)

for epoch in range(MAX_EPOCH):
    optimizer.zero_grad()
    running_loss = 0.0
    encoded_train = net_final.forward(X_train)  
    predict_train = net_final.classifier(encoded_train)  
    loss = criterion(predict_train, y_train)  
    loss.backward(retain_graph=True) 
    optimizer.step()  
    running_loss += loss.item()
    if epoch % VALID_FREQ == 0:
        print('final model, train set %d epoch, loss: %.6f' % (epoch, running_loss))

    if epoch % VALID_FREQ == 0:
        encoded_val = net_final.forward(X_val)  
        predict_val = net_final.classifier(encoded_val)  
        loss_val = criterion(predict_val, y_val)
        print('final model, validation set %d epoch, loss: %.6f' % (epoch, loss_val.item()))

        if loss_val.item() >= val_best_loss:
            fail_time += 1
        else:
            fail_time = 0
            val_best_loss = loss_val.item()

        if fail_time >= VALID_PATIENCE:
            val_best_loss_per_layer.append(val_best_loss) 
            break

correct = 0  
with torch.no_grad():
    pred_test_total = torch.zeros(X_test.shape[0], dtype=torch.int64) 
    if torch.cuda.is_available():
        block_num = int(np.ceil(X_test.shape[0] / TEST_SET_BATCH_SIZE)) 
        for i in range(block_num):
            if i < (block_num - 1):
                X_test_batch = copy.deepcopy(X_test[i * TEST_SET_BATCH_SIZE: (i + 1) * TEST_SET_BATCH_SIZE, :])
                X_test_batch = X_test_batch.cuda()
                encoded_test = net_final.forward(X_test_batch) 
                predict_test = net_final.classifier(encoded_test)  
                _, predict_test = torch.max(predict_test, 1)  
                pred_test_total[i * TEST_SET_BATCH_SIZE: (i + 1) * TEST_SET_BATCH_SIZE] = copy.deepcopy(predict_test.detach())
                del X_test_batch
            else:
                X_test_batch = copy.deepcopy(X_test[i * TEST_SET_BATCH_SIZE: -1, :])
                X_test_batch = X_test_batch.cuda()
                encoded_test = net_final.forward(X_test_batch)  
                predict_test = net_final.classifier(encoded_test)  
                _, predict_test = torch.max(predict_test, 1) 
                pred_test_total[i * TEST_SET_BATCH_SIZE: -1] = copy.deepcopy(predict_test.detach())
                del X_test_batch

    correct += (pred_test_total == y_test).sum().item()
    print('Accuracy of the network on the test samples: %.6f %%' % (100 * float(correct) / y_test.shape[0]))

    conf_matrix = confusion_matrix(y_test, pred_test_total)
    kappa = kappa_coefficient(conf_matrix)

MOEAD_end_time_0 = time.time()
time_0 = MOEAD_end_time_0 - MOEAD_start_time_0

print('kappa coefficient: ', kappa)

print('hidden_dim_list: \n', hidden_dim_list)
print('activation_func_list: \n', activation_func_list)

pred_test_total = np.array(pred_test_total)
y_test = np.array(y_test)
pred_DF = pd.DataFrame(pred_test_total)
y_test_DF = pd.DataFrame(y_test)
pred_DF.to_csv('predict_test_DF.csv')
y_test_DF.to_csv('y_test_DF.csv')

torch.save(net_final.state_dict(), 'stacked_AE_MOEA_D.zip')
print('net_final:\n', net_final)
hidden_dim_list_array = np.array(hidden_dim_list)
activation_func_list_array = np.array(activation_func_list)
np.savetxt('hidden_dim_list.csv', hidden_dim_list_array, delimiter=',')
np.savetxt('activation_func_list.csv', activation_func_list_array, delimiter=',')


del X_train, X_val, X_test
del y_train, y_val, y_test
del pred_test_total, pred_DF, y_test_DF

MOEAD_start_time_1 = time.time()
# =========================== predict the whole image =============================
data_total_combine = pd.read_csv('E:\\PythonProject\\StackedAE_MOEAD\\SanFrancisco2\\filtered_with_7x7_refined_Lee\\data_total_combine_5x5_SanFrancisco2.csv', header=None)    
scaler.transform(data_total_combine)  
data_total_combine = torch.FloatTensor(data_total_combine) 

with torch.no_grad():
    pred_whole_pic_total = torch.zeros(data_total_combine.shape[0], dtype=torch.int64)  
    if torch.cuda.is_available():
        block_num = int(np.ceil(data_total_combine.shape[0] / WHOLE_PIC_BATCH_SIZE)) 
        for i in range(block_num):
            if i < (block_num - 1):
                data_total_combine_batch = copy.deepcopy(data_total_combine[i * WHOLE_PIC_BATCH_SIZE: (i + 1) * WHOLE_PIC_BATCH_SIZE, :])
                data_total_combine_batch = data_total_combine_batch.cuda()
                data_total_combine_encoded = net_final.forward(data_total_combine_batch) 
                pred_total_batch = net_final.classifier(data_total_combine_encoded)  
                _, pred_total_batch = torch.max(pred_total_batch, 1) 
                pred_whole_pic_total[i * WHOLE_PIC_BATCH_SIZE: (i + 1) * WHOLE_PIC_BATCH_SIZE] = copy.deepcopy(
                    pred_total_batch.detach())
                del data_total_combine_batch
            else:
                data_total_combine_batch = copy.deepcopy(data_total_combine[i * WHOLE_PIC_BATCH_SIZE: -1, :])
                data_total_combine_batch = data_total_combine_batch.cuda()
                data_total_combine_encoded = net_final.forward(data_total_combine_batch)
                pred_total_batch = net_final.classifier(data_total_combine_encoded) 
                _, pred_total_batch = torch.max(pred_total_batch, 1) 
                pred_whole_pic_total[i * WHOLE_PIC_BATCH_SIZE: -1] = copy.deepcopy(pred_total_batch.detach())
                del data_total_combine_batch

pred_whole_pic_total = pred_whole_pic_total.reshape( [HEIGHT-4, WIDTH-4] ) 

MOEAD_end_time_1 = time.time()
time_1 = MOEAD_end_time_1 - MOEAD_start_time_1

pred_whole_pic_total = np.array(pred_whole_pic_total)       
pred_whole_pic_total_DF = pd.DataFrame(pred_whole_pic_total)
pred_whole_pic_total_DF.to_csv('pred_whole_pic_total.csv', index=False, header=False)

# San Francisco2 color
color = [[60,90,160], [0,190,0], [255,0,0], [102,102,0], [204,102,0]]     

pred_total_color_edition = np.zeros([HEIGHT-4, WIDTH-4, 3]) 

for i in range(HEIGHT-4):
    for j in range(WIDTH-4):
        if pred_whole_pic_total[i, j] == 0:
            pred_total_color_edition[i, j, :] = color[0]
        elif pred_whole_pic_total[i, j] == 1:
            pred_total_color_edition[i, j, :] = color[1]
        elif pred_whole_pic_total[i, j] == 2:
            pred_total_color_edition[i, j, :] = color[2]
        elif pred_whole_pic_total[i, j] == 3:
            pred_total_color_edition[i, j, :] = color[3]
        elif pred_whole_pic_total[i, j] == 4:
            pred_total_color_edition[i, j, :] = color[4]
        # elif pred_whole_pic_total[i, j] == 5:
            # pred_total_color_edition[i, j, :] = color[5]


pred_total_color_edition /= 255

program_end_time = time.time()
print('the running time of the whole program： %f s' % (program_end_time-program_start_time))

plt.imshow(pred_total_color_edition)
plt.axis('off')
plt.savefig('predict_result.tiff')
plt.show()










