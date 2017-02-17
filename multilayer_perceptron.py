'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate data
#============================
# Number of samples
size = 10000
# Consider integers in range (0, max_num)
max_num = 10
# Generate samples
train_data_x = np.random.random_integers(0, max_num, (size, 2))
# Simulate the addition operator
train_data_y = (train_data_x[:,0] + train_data_x[:,1]).reshape(size,1)

# Parameters
learning_rate = 0.01  # smaller learning rate results in too slow convergence
training_epochs = 200
batch_size = 100
display_step = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
# Input - two numbers to be added
n_input = 2
# Output - sum
n_output = 1

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.sigmoid(layer_1)
    # layer_1 = tf.nn.tanh(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation (ReLU activation leads to a terrible performance)
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.pow(y - pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Plot
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.title('Training phase')

# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #with tf.device("/cpu:0"):
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_data_x.shape[0]/batch_size)
        index = 0
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train_data_x[index:index+batch_size,:], train_data_y[index:index+batch_size]
            index += batch_size
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        plt.plot(epoch, avg_cost,'*')
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    # plt.show()


    #=============================================
    # Test model
    test_set_size = 100
    # Test addition operator in the same range of integers (0, max_num)
    # Generate data
    test_data_x = np.random.random_integers(0, max_num, (test_set_size, 2))
    test_data_y = (test_data_x[:, 0] + test_data_x[:, 1]).reshape(test_set_size, 1)
    # Calculate accuracy
    correct_prediction_int = tf.equal(tf.to_int32(pred), tf.to_int32(y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction_int, "float"))
    print("Accuracy in the same range: %.2f%%" % (accuracy.eval({x: test_data_x, y: test_data_y}) * 100))
    # Print some predictions
    prediction = pred.eval(feed_dict={x: test_data_x[:20]})
    print(prediction - test_data_y[:20])

    # correct_prediction = tf.abs(pred - y)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy in the same range: %.2f%%" %(accuracy.eval({x: test_data_x, y: test_data_y})*100))


    # Test in a different range of integers
    # Generate test data
    test_data_x = np.random.random_integers(0, max_num*max_num, (test_set_size, 2))
    test_data_y = (test_data_x[:, 0] + test_data_x[:, 1]).reshape(test_set_size, 1)
    # Calculate accuracy
    correct_prediction = tf.equal(tf.to_int32(pred), tf.to_int32(y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy in bigger range: %.2f%%" %(accuracy.eval({x: test_data_x, y: test_data_y})*100))

    # Display training phase
    plt.draw()
    plt.waitforbuttonpress()



