# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 21:33:48 2018

@author: LeeJY
"""

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split #DataSet에서 Training Data와 Test Data를 분류하기 위한 모듈 추가

zoo = pd.read_csv("./data/zoo.csv") #Kaggle에서 다운받은 csv형식의 파일 읽어오기

#zoo.csv의 마지막 열(column)인 class_type은 제외하고 animal_data 변수에 저장
animal_data = zoo.iloc[:,:-1] 

#zoo.csv의 마지막 열(column)인 class_type만 label_data 변수에 저장
label_data = zoo.iloc[:,-1:] 

#전체 DataSet에서 30%를 Test Data로 사용하며, 선택은 무작위로 함
train_x, test_x, train_y, test_y = train_test_split(animal_data,label_data,test_size=0.3,random_state=42,stratify=label_data)
 
#train_x, test_x의 가장 첫 번째 animal_name 열을 train_name, test_name 변수에 저장
train_name = train_x['animal_name']
test_name = test_x['animal_name']

#train_x, test_x의 가장 첫 번째 animal_name 열을 제외하고 숫자 값만 갖는 데이터 형식으로 만듬 
train_x = train_x.iloc[:,1:]
test_x = test_x.iloc[:,1:]

X = tf.placeholder(dtype=tf.float32, shape=[None,16],name="X") #Input Layer 노드의 수 : 16

Y = tf.placeholder(dtype=tf.int32, shape=[None,1], name="Y") #Output Layer 노드의 수 : 1

Y_one_hot = tf.one_hot(Y, 7)  # one hot encoding으로 7개 Class로 분류
Y_one_hot = tf.reshape(Y_one_hot, [-1, 7])

W = tf.Variable(tf.random_normal([16, 7],seed=0), name='W')

b = tf.Variable(tf.random_normal([7],seed=0), name='b')

logit = tf.matmul(X,W)+b 

hypothesis = tf.nn.softmax(logit)

cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,labels=Y_one_hot)

cost = tf.reduce_mean(cost_i)

train  = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5001):
        sess.run(train, feed_dict={X: train_x, Y: train_y})
        if step % 1000 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: train_x, Y: train_y})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
            
    train_acc = sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
    test_acc,test_predict,test_correct = sess.run([accuracy,prediction,correct_prediction], feed_dict={X: test_x, Y: test_y})
    print("Model Prediction =", train_acc)
    print("Test Prediction =", test_acc)
    
sub = pd.DataFrame()
sub['Name'] = test_name
sub['Predict_Type'] = test_predict
sub['Origin_Type'] = test_y
sub['Correct'] = test_correct

print(sub)