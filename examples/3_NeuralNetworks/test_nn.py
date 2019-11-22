""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import time
# Evaluate model
# Initialize the variables (i.e. assign their default value)

# Start training
with tf.Session() as sess:
    # Run the initializer
    saver = tf.train.import_meta_graph('my_test_model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    
    graph = tf.get_default_graph()
    XX=graph.get_tensor_by_name('X:0')
    YY=graph.get_tensor_by_name('Y:0')
    logits=graph.get_tensor_by_name('network:0')
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(YY, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    CNT = 10
    total_time = 0
    for i in range(CNT):
        start_time = time.time()
        sess.run(prediction, feed_dict={XX: mnist.test.images})
        t = time.time()
        print("Duration {} ms i = {}".format((t - start_time)*1000,i))
        total_time += (t-start_time)*1000

    print("aveage duration {} ms".format((total_time)/CNT))
