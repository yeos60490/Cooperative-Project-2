import tensorflow as tf
import numpy as np
import os
import csv

learning_rate = 0.001
directory = '/Users/Desktop/data'
continent = ['asia', 'europe', 'southamerica', 'northamerica', 'africa', 'australia']
con_len= len(continent )

train={}
x_data={}
y_data={}

for i in range (0,con_len):
    con = continent[i]
    train[con]= np.loadtxt(directory + '/train/' + continent[i] +  '_train.txt', unpack=True, dtype='float32')
    x_data[con] = np.transpose(train[con][0:44])
    y_data[con] = np.transpose(train[con][44:])

test_data={}
test_data_output={}

test = np.loadtxt(directory + '/test.txt', unpack=True, dtype='float32')
t_con = np.transpose(test[0: 6])
t_data = np.transpose(test[6: 50])
t_data_output = np.transpose(test[50:])

for j in range(0,6):
    a =np.array([])
    c =np.array([])
    con = continent[j]
    for i in range(len(t_con)):
        if t_con[i][j] == 1. :
            b = a
            a = np.append(b,t_data[i])
            d = c
            c = np.append(d,t_data_output[i])
    a=a.reshape(len(a)/44, 44)
    c=c.reshape(len(c),1)
    test_data[con]=a
    test_data_output[con]=c

classes = []
file = open(directory + '/class_name.txt', 'r') 
data = csv.reader(file)
for line in data:
    for one in line:
        classes.append(one)


X = tf.placeholder('float', [None, 44])
Y = tf.placeholder(tf.int32 , [None, 1])
Y_onehot= tf.one_hot(Y, len(classes))
Y_onehot= tf.reshape(Y_onehot, [-1,len(classes)])

W1 = tf.Variable(tf.random_normal([44, 44]))
W2 = tf.Variable(tf.random_normal([44, len(classes)]))

b1 = tf.Variable(tf.random_normal([44]))
b2 = tf.Variable(tf.random_normal([len(classes)]))

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1),b1))
hypothesis = tf.add(tf.matmul(L1, W2), b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y_onehot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()


#k=0~6
for k in range (len(test_data)):
    #print('----------------------',continent[k])
    con = continent[k]
    if(len(test_data[con])) !=0:
        with tf.Session() as sess:
            sess.run(init)

            for step in xrange(2001):
                sess.run(train, feed_dict={X: x_data[con], Y: y_data[con]})
                if step % 200 == 0:
                    print step, sess.run(cost, feed_dict={X: x_data[con], Y: y_data[con]})

            total=0

            for i in range(len(test_data[con])):
                #print (i)
                a = sess.run(hypothesis, feed_dict={X: [test_data[con][i]]})
                #print("original output: " + classes[int(test_data_output[con][i])])
                for j in range(0,3):
                    #print(a)
                    name = sess.run(tf.argmax(a, 1))
                    if(name==test_data_output[con][i]):
                        total=total+1
                    a[0][name] = -1
                    print (classes[int(name)])
            #print("total", total)
            #print ("len", len(test_data[con]))
            acc = float(total)/ float(len(test_data[con]))
            #print(con, "acc", acc*100)

