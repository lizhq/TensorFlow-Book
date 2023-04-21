# -*- coding: utf-8 -*-

# __ author:Jack
# date: 2023-04-20


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)


np.random.seed(100)

learning_rate = 0.001
training_epochs = 1000
reg_lambda = 0.


def split_dataset(x_dataset, y_dataset, ratio):
    arr = np.arange(x_dataset.size)
    np.random.shuffle(arr)
    num_train = int(ratio * x_dataset.size)
    x_train = x_dataset[arr[0:num_train]]
    y_train = y_dataset[arr[0:num_train]]
    x_test = x_dataset[arr[num_train:x_dataset.size]]
    y_test = y_dataset[arr[num_train:x_dataset.size]]
    return x_train, x_test, y_train, y_test

x_dataset = np.linspace(-1, 1, 100)

num_coeffs = 9
y_dataset_params = [0.] * num_coeffs
y_dataset_params[2] = 1
y_dataset = 0

for i in range(num_coeffs):
    y_dataset += y_dataset_params[i] * np.power(x_dataset, i)
y_dataset += np.random.randn(*x_dataset.shape) * 0.3

(x_train, x_test, y_train, y_test) = split_dataset(x_dataset, y_dataset, 0.7)


def model_polynomial(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.cast(tf.pow(X, i),dtype=tf.float32))
        terms.append(term)    
    return  tf.add_n(terms)

def polynomial():
    
    #plt.scatter(x_train, y_train)
    #plt.show()
    
    optimizer = tf.optimizers.Adam(0.01)
    w = tf.Variable([0.] * num_coeffs, name="parameters",dtype=tf.float32)

    for reg_lambda in np.linspace(0,1,100):
        for epoch in range(training_epochs):
            with tf.GradientTape() as tape:
                y_predict = model_polynomial(x_train,w)

                cost = tf.add(tf.reduce_sum(tf.square(y_train-y_predict)),tf.multiply(reg_lambda, tf.reduce_sum(tf.square(w))))
                cost = cost/2*x_train.size
        gradients = tape.gradient(cost,w)
        optimizer.apply_gradients(zip(gradients, w))
        
        print('reg lambda', reg_lambda)
        print('final cost', cost)


    print(w)
    plt.scatter(x_train, y_train)
    trY2 = 0
    for i in range(num_coeffs):
        trY2 += w[i] * np.power(x_train, i)
    plt.plot(x_train, trY2, 'r')
    plt.show()

'''
main
'''
def main():
    polynomial()
    print('===')

if __name__=='__main__':
    main()