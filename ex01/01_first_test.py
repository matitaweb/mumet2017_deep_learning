import tensorflow as tf
import numpy as np

print(tf.__version__)   

#Esempio di regressione

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3 + np.random.normal(0.0, 0.05)

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)

#vettore pesi VARIABILE a ogni interazione con valori compresi tra -1 e +1 
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# vettore bias paro da zero
b = tf.Variable(tf.zeros([1]))
# costruisco la del risultato
# y = W * x_data + b
y = tf.add(tf.multiply(x_data, W), b)

# creo la funzione di riduzione perdita con (scarto quadratico medio tra la predizione e il dato vero)
loss = tf.reduce_mean(tf.square(y - y_data))

# creo un ottimizzatore 
optimizer = tf.train.GradientDescentOptimizer(0.5)

# inserisco la funzione di riduzione chiedendo al'ottimizzatore di minimizzarla
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(2001):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(W), sess.run(b))