import tensorflow as tf
import pandas as pd

from animal_classification_model import animal_classifier
from sklearn.model_selection import train_test_split

zoo = pd.read_csv("./data/zoo.csv")

animal_data = zoo.iloc[:,:-1]

label_data = zoo.iloc[:,-1:]

train_x, test_x, train_y, test_y = train_test_split(animal_data,label_data,test_size=0.3,random_state=42,stratify=label_data)

train_name = train_x['animal_name']
test_name = test_x['animal_name']

train_x = train_x.iloc[:,1:]
test_x = test_x.iloc[:,1:]

sess = tf.Session()

model = animal_classifier(sess) #객체 생성 

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

for step in range(5001):
    a_ = model._train_model(train_x,train_y)
    if step % 1000 == 0:
        print("Step: {:5}\tAcc: {:.2%}".format(step, a_))

saver.save(sess,"./models/",global_step=5001)
