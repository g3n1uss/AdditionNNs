'''

'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

# Generate data
def generate_data(start, end, size):
    # Generate samples
    data_x = np.random.random_integers(start, end, (size, 2))
    # Simulate the addition operator
    data_y = (data_x[:, 0] + data_x[:, 1]).reshape(size, 1)
    return data_x, data_y

# Number of samples
size = 10000
# Consider integers in range (0, max_num)
max_num = 10
# Generate samples
train_data_x, train_data_y = generate_data(0, max_num, size)

# Parameters
learning_rate = 0.01  # smaller learning rate results in too slow convergence
training_epochs = 500
batch_size = 100
display_step = 10

# Input - two numbers to be added
n_input = 2
# Output - sum
n_output = 1

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

weights = {'w1':tf.Variable(tf.random_normal([n_input, n_output]))}
biases = {'b1':tf.Variable(tf.random_normal([n_output]))}
out = tf.add(tf.matmul(x, weights['w1']), biases['b1'])

# Build a model
# 'out' gives predictions

# Define loss and optimizer
cost = tf.reduce_mean(tf.pow(y - out, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.round(out), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    #with tf.device("/cpu:0"):
    sess.run(init)
    print('Bias before training is:', biases['b1'].eval())
    print('Weights before training are:', weights['w1'].eval())
    # Training cycle
    for epoch in range(training_epochs):
        # Total number of batches
        total_batch = int(train_data_x.shape[0]/batch_size)

        # Loop over all batches
        index = 0
        for i in range(total_batch):
            batch_x, batch_y = train_data_x[index:index+batch_size,:], train_data_y[index:index+batch_size]
            index += batch_size
            # Run optimization op (backprop) and cost op (to get cost value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
    print("\rOptimization Finished!")

    print('Bias after trining is:', biases['b1'].eval())
    print('Weights after training are:', weights['w1'].eval())

    # Test model
    test_set_size = 1000
    # Test the addition operator in the same range of integers (0, max_num)
    test_data_x, test_data_y = generate_data(0, max_num, test_set_size)
    # Calculate accuracy
    print("Accuracy in the same range: %.2f%%" % (accuracy.eval({x: test_data_x, y: test_data_y}) * 100))

    # Test in a different range of integers (generalizability)
    test_data_x, test_data_y = generate_data(max_num, max_num*max_num, test_set_size)
    # Calculate accuracy
    print("Accuracy in a bigger range: %.2f%%" % (accuracy.eval({x: test_data_x, y: test_data_y}) * 100))
    # At this point we see that predictions are pretty bad

    #===================================================================
    # Manual testing
    tmp = np.array([[1235, 434]])
    print(" Prediction for %s is %.2f" % (
        str(tmp[0, 0]) + '+' + str(tmp[0, 1]) + '=' + str(tmp[0, 0] + tmp[0, 1]), out.eval(feed_dict={x: tmp})))
    weightsmtr = weights['w1'].eval()
    biasesmtr = biases['b1'].eval()
    #===================================================
    # Now let's try to substitute the known solution, weights are set to ones and bias is zero
    # this would give exactly the addition operator

    # This is how we assing values to weights in tensorflow
    weights['w1'].assign([[1], [1]]).eval()
    biases['b1'].assign([0]).eval()

    # Print weights and biases
    print('Bias after trining is:', biases['b1'].eval())
    print('Weights after training are:', weights['w1'].eval())

    # Now make a prediction
    print(" Prediction for %s is %.2f" % (
    str(tmp[0, 0]) + '+' + str(tmp[0, 1]) + '=' + str(tmp[0, 0] + tmp[0, 1]), out.eval(feed_dict={x: tmp})))
    # We see that it works and we get exactly what we wanted

    #=========================================================
    # So the conclusion is that it is impossible to train even a perceptron to perform addition because we
    # can come at close to weights 1,1 as we want, but we never get exactly 1,1, which results in bad performance
    # for large values of inputs







