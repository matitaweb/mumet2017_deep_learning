import tensorflow as tf
import numpy as np
from lab_utils import get_brain_body_data

print("TF versions: " + tf.__version__) 


# 1. Load data using lab utils/get brain body data
body_weight, brain_weight = get_brain_body_data("data/brain_body_weight.txt")
print("body_weight len " + str(len(body_weight)))
print("brain_weight len " + str(len(brain_weight)))
n_samples = len(body_weight)
print("samples: len " + str(n_samples))

# 2. Define appropriate placeholders using tf.placeholder
# Define placeholders (1-d)
x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

# 3. Define weight and bias variables using tf.Variable
# Define variables
w = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))


#4. Define 1-d linear regression model ypred = xw + b
# Linear regression model
y_pred = tf.add(tf.multiply(x, w), b)

# 5. Define an appropriate objective function, e.g. (ypred − ytrue ) al quadrato
# Define objective function
loss = tf.reduce_mean(tf.square(y_pred - y))

# 6. Create an Optimizer (e.g. tf.train.AdamOptimizer)
# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.005)

# 7. Define a train iteration as one step of loss minimization
# Define one training iteration
train_step = optimizer.minimize(loss)

# 8. Loop train iteration until convergence
n_epochs = 100

with tf.Session() as sess:
    # Initialize all variables
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(n_epochs):
        total_loss = 0
        for bo_w, br_w in zip(body_weight, brain_weight):
            t, l = sess.run([train_step, loss], feed_dict={x: bo_w, y: br_w})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    print('y = {0} * x + {1}'.format(sess.run(w), sess.run(b)))

    
    

