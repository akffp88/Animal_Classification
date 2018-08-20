import tensorflow as tf

import pandas as pd

from sklearn.model_selection import train_test_split

from animal_classification_model import animal_classifier


zoo = pd.read_csv("./data/zoo.csv")

animal_data = zoo.iloc[:,:-1]

label_data = zoo.iloc[:,-1:]

train_x, test_x, train_y, test_y = train_test_split(animal_data,label_data,test_size=0.3,random_state=42,stratify=label_data)

train_name = train_x['animal_name']
test_name = test_x['animal_name']

train_x = train_x.iloc[:,1:]
test_x = test_x.iloc[:,1:]

sess = tf.Session()

model = animal_classifier(sess) 

saver = tf.train.Saver()

saver.restore(sess,tf.train.latest_checkpoint('./models/'))

prediction_result = model._prediction_(test_x,test_y)

prediction_result = model._prediction_(test_x,test_y)

sub = pd.DataFrame()
sub['Animal_name'] = test_name
sub['Predict_Type'] = prediction_result
sub['Origin_Type'] = test_y

print(sub)
