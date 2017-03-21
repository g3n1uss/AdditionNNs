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
training_epochs = 100
batch_size = 100
display_step = 10

# Input - two numbers to be added
n_input = 2
# Output - sum
n_output = 1

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

# Build a model
def define_model(x, y, layers):
    weights, biases = {}, {}
    for i in range(1, len(layers)):
        weights['w'+str(i)] = tf.Variable(tf.random_normal([layers[i - 1], layers[i]]))
        biases['b'+str(i)] = tf.Variable(tf.random_normal([layers[i]]))
    out = x
    for i in range(1, len(weights) + 1):
        out = tf.add(tf.matmul(out, weights['w'+str(i)]), biases['b'+str(i)])
        # Do not add relu in the end, it leads to a terrible performance
        if i < len(weights):
            out = tf.nn.relu(out)
    # 'out' gives predictions

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.pow(y - out, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    # Evaluate model
    correct_pred = tf.equal(tf.round(out), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
    return [out, weights, biases, cost, optimizer, accuracy]

#===========================================
# Multilayer non-linear neural net
#===========================================
# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
mlp = define_model(x, y, [n_input, n_hidden_1, n_hidden_2, n_output])
#=================================
# Linear perceptron
#=================================
p = define_model(x, y, [n_input, n_output])
#===============================================================

# All models
models = {'p':{'pred':p[0], 'cost':p[3], 'optimizer':p[4], 'accuracy':p[5]},
          'mlp':{'pred':mlp[0], 'cost':mlp[3], 'optimizer':mlp[4], 'accuracy':mlp[5]}}
# Map
model_names_map = {'p':'Linear perceptron', 'mlp':'Multilayer ReLU neural net'}
# Average costs along training here
avr_costs = {}
for j in models.keys():
    avr_costs[j] = [0]*training_epochs

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    #with tf.device("/cpu:0"):
    sess.run(init)
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
            for j in models.keys():
                _, c = sess.run([models[j]['optimizer'], models[j]['cost']], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avr_costs[j][epoch] += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), end=' ')
            for j in models.keys():
                print(" cost "+j+"=", "{:.9f}".format(avr_costs[j][epoch]))
    print("\rOptimization Finished!")

    # Test model
    test_set_size = 1000
    # Test the addition operator in the same range of integers (0, max_num)
    test_data_x, test_data_y = generate_data(0, max_num, test_set_size)
    # Calculate accuracy
    for j in models.keys():
        print("Accuracy in the same range %s: %.2f%%" % (j, models[j]['accuracy'].eval({x: test_data_x, y: test_data_y}) * 100))

    # Print some predictions
    # prediction = np.rint(pred.eval(feed_dict={x: test_data_x}))
    # Numpy alternative to tensorflow calculations
    # print("Test prediction from numpy:", (1-np.sum(np.absolute(prediction - test_data_y))/test_data_y.shape[0])*100)

    # Test in a different range of integers (generalizability)
    test_data_x, test_data_y = generate_data(max_num, max_num*max_num, test_set_size)
    # Calculate accuracy
    for j in models.keys():
        print("Accuracy in a bigger range %s: %.2f%%" % (j, models[j]['accuracy'].eval({x: test_data_x, y: test_data_y}) * 100))

    # Display the training phase

    # Change title of the figure window
    fig = plt.gcf()
    fig.canvas.set_window_title('Training non-linear MLP and linear perceptron')
    for j in avr_costs.keys():
        plt.plot(np.array(range(epoch + 1)), np.log(np.array(avr_costs[j])), 'o', label=model_names_map[j])
    plt.ylabel('Log(Cost)')
    plt.xlabel('Epoch')
    plt.title('Training phase')
    plt.legend(loc='upper right')
    plt.draw()
    plt.waitforbuttonpress()



