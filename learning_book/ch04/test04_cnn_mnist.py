import tensorflow as tf
print('TensorFlow:{}'.format(tf.__version__))
import numpy as np
print('NumPy:{}'.format(np.__version__))
import matplotlib.pyplot as plt
import matplotlib
print('Matplotlib:{}'.format(matplotlib.__version__))

from tensorflow.examples.tutorials.mnist import input_data

from learning_book.ch04.test04_common import *



x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 将图像转化为2D图像格式, 28x28x1
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 5x5卷积
conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
conv1_pool = max_pool_2x2(conv1)

# 64个特征
conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

# 在两个卷积层和池化层后的图像大小是7x7x64， 原始的28x28像素的图片被降低至14x14， 然后在两个池化操作后变成7x7
# 64指的是第二个卷积层创建的特征图的数目
conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
# 1024个单元的全连接层
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)


DATA_DIR = '../data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

STEPS = 1000

with tf.Session() as tfs:
    tfs.run(tf.global_variables_initializer())
    for i in range(STEPS):
        batch = mnist.train.next_batch(50)
        tfs.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        if i % 100 == 0:
            train_accuracy = tfs.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step {}, training accuracy {}'.format(i, train_accuracy))

    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)
    test_accuracy = np.mean([tfs.run(accuracy,
                                     feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0}) for i in range(10)])

print('test accuracy: {}'.format(test_accuracy))
