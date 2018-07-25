# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:50:08 2018

@author: LEE
"""

import tensorflow as tf
import sys
import os
from animal_classification_model import animal_classifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

zoo = pd.read_csv("./data/test.csv")

animal_data = zoo.iloc[:,:-1]

label_data = zoo.iloc[:,-1:]

model = dict()

def _setup_():
    global model, animal_data, label_data
    
    sess = tf.Session()
    
    model = animal_classifier(sess)
    
    saver = tf.train.Saver()
    
    saver.restore(sess,tf.train.latest_checkpoint('./models'))
    
def _get_response_(content):
    if content == "Bear":
        test_data = animal_data.iloc[0,1:]
        test_label = label_data.iloc[0,:]
       
        test_data = pd.Series.to_frame(test_data)
        test_data = pd.DataFrame.transpose(test_data)

        test_label = pd.Series.to_frame(test_label)
        test_label = pd.DataFrame.transpose(test_label)

        result = model._prediction_(test_data,test_label)
        
    elif content == "Dolphin":
        test_data = animal_data.iloc[1,1:]
        test_label = label_data.iloc[1,:]
        
        test_data = pd.Series.to_frame(test_data)
        test_data = pd.DataFrame.transpose(test_data)

        test_label = pd.Series.to_frame(test_label)
        test_label = pd.DataFrame.transpose(test_label)

        result = model._prediction_(test_data,test_label)
        
    elif content == "Duck":
        test_data = animal_data.iloc[2,1:]
        test_label = label_data.iloc[2,:]
        
        test_data = pd.Series.to_frame(test_data)
        test_data = pd.DataFrame.transpose(test_data)

        test_label = pd.Series.to_frame(test_label)
        test_label = pd.DataFrame.transpose(test_label)

        result = model._prediction_(test_data,test_label)
        
    elif content == "Elephant":
        test_data = animal_data.iloc[3,1:]
        test_label = label_data.iloc[3,:]

        test_data = pd.Series.to_frame(test_data)
        test_data = pd.DataFrame.transpose(test_data)

        test_label = pd.Series.to_frame(test_label)
        test_label = pd.DataFrame.transpose(test_label)

        result = model._prediction_(test_data,test_label)
        
    elif content == "Frog":
        test_data = animal_data.iloc[4,1:]
        test_label = label_data.iloc[4,:]
        
        test_data = pd.Series.to_frame(test_data)
        test_data = pd.DataFrame.transpose(test_data)

        test_label = pd.Series.to_frame(test_label)

        result = model._prediction_(test_data,test_label)
    
    elif content == "Gorilla":
        test_data = animal_data.iloc[5,1:]
        test_label = label_data.iloc[5,:]
        
        test_data = pd.Series.to_frame(test_data)
        test_data = pd.DataFrame.transpose(test_data)

        test_label = pd.Series.to_frame(test_label)        

        result = model._prediction_(test_data,test_label)
    
    elif content == "Honeybee":
        test_data = animal_data.iloc[6,1:]
        test_label = label_data.iloc[6,:]
        
        test_data = pd.Series.to_frame(test_data)
        test_data = pd.DataFrame.transpose(test_data)

        test_label = pd.Series.to_frame(test_label)        

        result = model._prediction_(test_data,test_label)

    elif content == "Lobster":
        test_data = animal_data.iloc[7,1:]
        test_label = label_data.iloc[7,:]
        
        test_data = pd.Series.to_frame(test_data)
        test_data = pd.DataFrame.transpose(test_data)

        test_label = pd.Series.to_frame(test_label)        

        result = model._prediction_(test_data,test_label)

    elif content == "Octopus":
        test_data = animal_data.iloc[8,1:]
        test_label = label_data.iloc[8,:]
        
        test_data = pd.Series.to_frame(test_data)
        test_data = pd.DataFrame.transpose(test_data)

        test_label = pd.Series.to_frame(test_label)        

        result = model._prediction_(test_data,test_label)
    
    elif content == "Seahorse":
        test_data = animal_data.iloc[9,1:]
        test_label = label_data.iloc[9,:]
        
        test_data = pd.Series.to_frame(test_data)
        test_data = pd.DataFrame.transpose(test_data)

        test_label = pd.Series.to_frame(test_label)        

        result = model._prediction_(test_data,test_label)

    elif content == "Seasnake":
        test_data = animal_data.iloc[10,1:]
        test_label = label_data.iloc[10,:]
        
        test_data = pd.Series.to_frame(test_data)
        test_data = pd.DataFrame.transpose(test_data)

        test_label = pd.Series.to_frame(test_label)        

        result = model._prediction_(test_data,test_label)
    
    else :
        result = "올바른 값을 입력해주세요."

    return result