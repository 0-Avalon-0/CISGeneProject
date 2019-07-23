from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

np.set_printoptions(suppress=True)

test_data = pd.HDFStore('C:\dataset\\test_data.h5')
test_feature_matrix_dataframe = test_data['rpkm']
test_feature_matrix = test_feature_matrix_dataframe.values
# 训练矩阵行数，即样本个数
train_matrix_size = test_feature_matrix.shape[0]


# Visualize decoder setting
# Parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 256
display_step = 1
examples_to_show = 10

# encode train array
temp_matrix = np.zeros(shape=[2855, 512])
encode_array = tf.placeholder(dtype=tf.float32, shape=[2855, 512], name='encode_array')
encode_variable = tf.Variable(tf.zeros([2855, 512]), name='encode_variable')
encode = tf.assign(encode_variable, encode_array)

# Network Parameters

# n_input = 784  # MNIST data input (img shape: 28*28)
n_input = 20499
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# hidden layer settings
n_hidden_1 = 1024 # 1st layer num features
n_hidden_2 = 512 # 2nd layer num features
# n_hidden_1 = 9000
# n_hidden_2 = 1024
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# Building the encoder

def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

# Building the decoder

def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# define saver
var_dict = {'encode_variable': encode_variable}
saver = tf.train.Saver(var_dict)


# Launch the graph
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    total_batch = math.ceil(train_matrix_size/batch_size)

    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            batch_xs = test_feature_matrix[i * batch_size: (i + 1) * batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

            if epoch == training_epochs-1:
                if i == total_batch-1:
                    temp_matrix = (sess.run(encoder_op, feed_dict={X: test_feature_matrix}))
                    # print(temp_matrix)
                    print(temp_matrix.shape[0])
                    _ = sess.run(encode, feed_dict={encode_array: temp_matrix})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    save_path = saver.save(sess, 'C:\\Users\\11201\\Desktop\\parameter\\save_net.ckpt')
    print('Save to path: ', save_path)

