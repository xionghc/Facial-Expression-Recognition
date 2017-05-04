import numpy as np
import tensorflow as tf

from utils import *

import os

def deepnn(x):
  x_image = tf.reshape(x, [-1, 48, 48, 1])
  # conv1
  W_conv1 = weight_variables([5, 5, 1, 64])
  b_conv1 = bias_variable([64])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  # pool1
  h_pool1 = maxpool(h_conv1)
  # norm1
  norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
  
  # conv2
  W_conv2 = weight_variables([3, 3, 64, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
  h_pool2 = maxpool(norm2)

  # Fully connected layer
  W_fc1 = weight_variables([12*12*64, 384])
  b_fc1 = bias_variable([384])
  h_conv3_flat = tf.reshape(h_pool2, [-1, 12*12*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

  
  # Fully connected layer
  W_fc2 = weight_variables([384, 192])
  b_fc2 = bias_variable([192])
  h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

  # linear
  W_fc3 = weight_variables([192, 7])
  b_fc3 = bias_variable([7])
  y_conv = tf.add(tf.matmul(h_fc2, W_fc3), b_fc3)

  return y_conv


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variables(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def run_train_model(_):

  fer2013 = input_data('./data/fer2013/fer2013.csv')
  max_train_steps = 15000

  x = tf.placeholder(tf.float32, [None, 2304])
  y_ = tf.placeholder(tf.float32, [None, 7])

  y_conv =  deepnn(x)

  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for step in range(max_train_steps):
      batch = fer2013.train.next_batch(50)
      if step % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1]})
        print('step %d, training accuracy %g' % (step, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})

      if step+1 == max_train_steps:
        saver.save(sess, '/tmp/models/emotion_model', global_step = step+1)

    print('test accuracy %g' % accuracy.eval(feed_dict={
      x: fer2013.validation.images, y_: fer2013.validation.labels}))


def run_test_model(_):


  x = tf.placeholder(tf.float32, [None, 2304])
  y_conv = deepnn(x)

  saver = tf.train.Saver()

  probs = tf.nn.softmax(y_conv)
  y_ = tf.argmax(probs)

  with tf.Session() as sess:
    assert os.path.exists('/tmp/models/emotion_model')
    saver.restore(sess, '/tmp/models/emotion_model')
    probs = sess.run(probs, feed_dict={x: [255]*2304})
    print(y_)


if __name__ == '__main__':
    tf.app.run(main=run_train_model)