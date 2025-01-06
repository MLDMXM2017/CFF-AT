# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 11:45:19 2021
@author: GaoJie
Description：Code for "Multi-task Cascade Forest Framework for Predicting Acute Toxicity Across Species"
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.model_selection import train_test_split
from deepforest import CascadeForestRegressor
import sys
import time
from parameters import *
sys.path.insert(0, "../")
start_time=time.time()

#******parameter display******
test_data   = "[3]"
train_data1 = "[0]"
train_data2 = "[1]"
train_data3 = "[2]"
train_data4 = "[4]"
print("*****parameter seting*****")   
print("Feature type:", feature_type)
print("Num of subtasks:",end-1)
print("Seed:",seed)
print("Sequence of folds:", test_data, train_data1, train_data2, train_data3, train_data4)
print("Type of distance metric:", distance_type)

#******load data from all subtasks******
for i in range(start,end):
    path_feature='/home/xxx/Data/Feature/' + feature_type + '/Label_' + str(i) +'.csv' 
    #feature's loading path
    path_label='/home/xxx/Data/Label/Label_'+str(i)+'.csv' 
    #label's loading path
    sentence = "X"+str(i)+ "= pd.read_csv(path_feature)"
    exec(sentence)
    sentence = "X"+str(i)+ "= np.array(X"+str(i)+".values[:,1:])"
    exec(sentence)
    #load label
    sentence = "Y"+str(i)+ "= pd.read_csv(path_label)"
    exec(sentence)    
    sentence="Y"+str(i)+"= np.array(Y"+str(i)+".values[:,-1])"
    exec(sentence) 

#******split each subtask dataset to five folds******
def get_cross_validation_data(X,Y):    
    data=[];label=[]
    kf = KFold(n_splits=5,shuffle=True,random_state=seed)
    for train_index, test_index in kf.split(X):
        data.append(X[test_index])
        label.append(Y[test_index])        
    return data,label

for i in range(start,end):
    sentence="data"+str(i)+",label"+str(i)+"=get_cross_validation_data(X"+str(i)+",Y"+str(i)+")"
    exec(sentence)
    #print("data ",i," split successfully")

#******main experiment begins, divide training set and testing set
x_train, y_train, x_train_list, y_train_list,x_test_list,y_test_list=[],[],[],[],[],[] 
for i in range(start,end):    
    sentence="x_test_list.append(data"+str(i) + test_data +")"
    exec(sentence)    
    sentence="y_test_list.append(label"+str(i)+ test_data +")"
    exec(sentence)
    #The testing set is divided for the i-th subtask, accounting for 20% of entire samples 
    sentence="fold1_x=list(data"+str(i) + train_data1 + ")"
    exec(sentence)
    sentence="fold2_x=list(data"+str(i) + train_data2 + ")"
    exec(sentence)
    sentence="fold3_x=list(data"+str(i) + train_data3 + ")"
    exec(sentence)
    sentence="fold4_x=list(data"+str(i) + train_data4 + ")"
    exec(sentence)
    sentence="fold1_y=list(label"+str(i)+ train_data1 + ")"
    exec(sentence)
    sentence="fold2_y=list(label"+str(i)+ train_data2 + ")"
    exec(sentence)
    sentence="fold3_y=list(label"+str(i)+ train_data3 + ")"
    exec(sentence)
    sentence="fold4_y=list(label"+str(i)+ train_data4 + ")"
    exec(sentence)      
    x_temp=fold1_x + fold2_x + fold3_x + fold4_x
    y_temp=fold1_y + fold2_y + fold3_y + fold4_y
    #The remaining 80% samples of i-th subtask are merged as training set
    x_train=x_train + x_temp
    y_train=y_train + y_temp
    x_train_list.append(np.array(x_temp,dtype=float)) 
    y_train_list.append(y_temp)

#******extract deep forest's top layer as source domain model
x_train = np.array(x_train)
y_train = np.array(y_train)
print("Shape of entire training sets:",x_train.shape, y_train.shape)
source_model = CascadeForestRegressor(verbose=0, n_jobs=-1,random_state=seed)
source_model.fit(x_train,y_train)
#print("Initial layer num of deep forest:", source_model.n_layers_)
layer0 = source_model.layers_["layer_0"] 

#******transfer knowledge from source domain to target domain
def transfer_knowledge_to_specific_task(source_model, train_data, train_label, test_data, test_label,seed): 
    train_data = np.array(train_data, dtype="uint8")
    test_data = np.array(test_data, dtype="uint8")
    enhanced_feature1 = source_model.predict_full(train_data)
    enhanced_feature2 = source_model.predict_full(test_data)    
    train_data = np.concatenate([train_data, enhanced_feature1], axis=1)
    test_data = np.concatenate([test_data, enhanced_feature2], axis=1)    
    target_model = CascadeForestRegressor(verbose=0, n_jobs=-1,random_state=seed)
    target_model.fit(train_data, train_label)  
    y_pred = target_model.predict(test_data)
    r2 = r2_score(test_label,y_pred)
    mse = mean_squared_error(test_label,y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(test_label,y_pred)
    return r2,rmse,mae

task_relation=np.loadtxt(distance_type) #load the calculated covariance relationship among the subtasks
task_num=(len(x_train_list))
relation=task_relation[:task_num,:task_num]
r2_fold1=[]  #store r2 for 59 subtasks
rmse_fold1=[]
mae_fold1=[]

print("")
print("*****modeling for each subtask*****")  
for i in range(len(x_test_list)): 
    print(i+1,"-th task's initial training data shape:", np.array(x_train_list[i]).shape)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train_list[i], y_train_list[i], test_size=0.25, random_state=seed)
    base_r2,rmse,mae = transfer_knowledge_to_specific_task(layer0, x_train, y_train, x_validation, y_validation, seed=seed)
    rank = np.argsort(relation[i])
    add_num=0
    #x_temp, y_temp=list(x_train),list(y_train)
    for j in range(len(rank)):
        if rank[j]==i: continue
        else:
            x_temp = list(x_train) + list(x_train_list[rank[j]])
            y_temp = list(y_train) + list(y_train_list[rank[j]])
            #print(np.array(x_temp).shape)
            #print(np.array(y_temp).shape)
            temp_r2,rmse,mae = transfer_knowledge_to_specific_task(layer0, x_temp, y_temp, x_validation, y_validation,seed=seed)
            if temp_r2 > base_r2: #execute greedy neighbor retrieval strategy
                base_r2 = temp_r2
                x_train = x_temp
                y_train = y_temp
                add_num = add_num + 1
            else: break
    
    x = np.array(list(x_train)+list(x_validation))
    y = np.array(list(y_train)+list(y_validation))   
    r2,rmse,mae = transfer_knowledge_to_specific_task(layer0, x, y, x_test_list[i], y_test_list[i],seed=seed) 
    #training and testing process for each subtask 
    r2_fold1.append(r2)
    rmse_fold1.append(rmse)
    mae_fold1.append(mae)
    print("r2:", round(r2,3))

print("")   
print("*****average performance for all subtasks*****")   
print("all subtask's average r2:", round(np.mean(r2_fold1),3))
print("all subtask's average rmse:", round(np.mean(rmse_fold1),3))
print("all subtask's average mae:", round(np.mean(mae_fold1),3))
end_time=time.time()
hours = (end_time-start_time)/(60*60)
print("time-consuming:", round(hours,3),"hours")