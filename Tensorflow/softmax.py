# coding=utf-8
# machine learning practice
# Yuliang Sun, 2016

# load data
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

# input
x = tf.placeholder("float", [None, 784])
# weight and bias
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# hypothesis
y = tf.nn.softmax(tf.matmul(x,W) + b)
# true value
y_ = tf.placeholder("float", [None,10])
# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# gradient descent
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# training
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# test
print "Accuracy is "
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
