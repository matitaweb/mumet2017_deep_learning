#https://www.tensorflow.org/get_started/mnist/beginners

import tensorflow as tf
import numpy as np
from lab_utils import get_mnist_data

print("TF versions: " + tf.__version__) 


def simple_net(x):

    # 3. Define weight and bias variables using tf.Variable
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    # 3. Define weight and bias variables using tf.Variable
    logits = tf.matmul(x, W) + b
    y = tf.nn.softmax(logits)
    return y
    


if __name__ == '__main__':
    # 1. Load data using lab utils/get brain body data
    mnist_data = get_mnist_data("./data/", one_hot=True, verbose=True)
    print("mnist_data: " + str(mnist_data))
    
    # 2. Define appropriate placeholders using tf.placeholder
    x  = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    # Create the model
    y = simple_net(x)
    
    
    # Define loss and optimizer
    eps = np.finfo("float32").eps
    loss = tf.reduce_mean(tf.reduce_sum(y_ * -1 * tf.log(y +eps), 1)) # acc=0,66 senza rediction_indices
    
    #cross_entropy_loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    optimizer = tf.train.GradientDescentOptimizer(0.5);
    #train_step = optimizer.minimize(cross_entropy_loss)
    train_step = optimizer.minimize(loss)
    
    # accuracy part
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.Session() as sess:
        
        # Initialize all variables
        init = tf.initialize_all_variables()
        sess.run(init)
        
        # Train
        batch_size =128
        n_epochs = 1001
        for i in range(n_epochs):
            batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
            t, l = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys})
            if i % 100 == 0:
                print('Epoch {0}, loss: {1}, acc: {2}'.format(i, l, sess.run(accuracy, feed_dict={x: mnist_data.test.images,  y_: mnist_data.test.labels})))

    
        # Test trained model
        
        #print(sess.run(accuracy, feed_dict={x: mnist_data.test.images,  y_: mnist_data.test.labels}))

