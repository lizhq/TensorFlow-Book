# -*- coding: utf-8 -*-

# __ author:Jack
# date: 2023-04-20


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)


learning_rate = 0.01
training_epochs = 40



def model(X, w):
    return tf.multiply(X, w)

'''
拟合 y = wx
'''
def simple_line():
    x_train = np.linspace(-1, 1, 101)
    y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33

    #plt.scatter(x_train, y_train)
    #plt.show()
    
    optimizer = tf.optimizers.Adam(0.01)
    w = tf.Variable(0.0, name="weights")

    for epoch in range(training_epochs):
        for (x, y) in zip(x_train, y_train):
            with tf.GradientTape() as tape:
                y_predict = model(x,w)
                cost = tf.reduce_mean(tf.square(y_predict-y))
            print(epoch,"cost", cost)

            gradients = tape.gradient(cost,[w])
            print(epoch,"grads", gradients)

            optimizer.apply_gradients(zip(gradients, [w]))

    print(w)

    plt.scatter(x_train, y_train)
    y_learned = x_train*w.numpy()
    plt.plot(x_train, y_learned, 'r')
    plt.show()


def simple_line_other():
    x_train = np.linspace(-1, 1, 101)
    y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33

    #plt.scatter(x_train, y_train)
    #plt.show()
    
    optimizer = tf.optimizers.Adam(0.01)
    w = tf.Variable(0.0, name="weights")

    batch_size = 20
    for epoch in range(training_epochs):
        for i in range(0,len(x_train),batch_size):
            start = i
            end = i + 20
            
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]

            loss = lambda: tf.losses.mean_squared_error(model(x_batch,w),y_batch)

            optimizer.minimize(loss,[w])
            print(epoch,"cost", loss)

    print(w)

    plt.scatter(x_train, y_train)
    y_learned = x_train*w.numpy()
    plt.plot(x_train, y_learned, 'r')
    plt.show()


trX = np.linspace(-1, 1, 101)
num_coeffs = 6
trY_coeffs = [1, 2, 3, 4, 5, 6]
trY = 0
for i in range(num_coeffs):
    trY += trY_coeffs[i] * np.power(trX, i)
trY += np.random.randn(*trX.shape) * 1.5

def model_polynomial(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.cast(tf.pow(X, i),dtype=tf.float32))
        terms.append(term)    
    return  tf.add_n(terms)

def simple_polynomial():
    
    
    #plt.scatter(x_train, y_train)
    #plt.show()
    
    optimizer = tf.optimizers.Adam(0.01)
    #w = tf.Variable([0.] * num_coeffs, name="parameters",dtype=tf.float32)
    w1 = tf.Variable(0.0, name="weights1")
    w2 = tf.Variable(0.0, name="weights2")
    w3 = tf.Variable(0.0, name="weights3")
    w4 = tf.Variable(0.0, name="weights4")
    w5 = tf.Variable(0.0, name="weights5")
    w6 = tf.Variable(0.0, name="weights6")
    w = [w1,w2,w3,w4,w5,w6]
    for epoch in range(training_epochs):
        for (x, y) in zip(trX, trY):     
            loss = lambda: tf.losses.mean_squared_error([model_polynomial(x,w)],[y])
            optimizer.minimize(loss,w)
            print(epoch,"loss", loss)

    print(w)
    plt.scatter(trX, trY)
    trY2 = 0
    for i in range(num_coeffs):
        trY2 += w[i] * np.power(trX, i)
    plt.plot(trX, trY2, 'r')
    plt.show()

'''
main
'''
def main():
    #simple_line()
    #simple_line_other()
    simple_polynomial()
    print('===')

if __name__=='__main__':
    main()