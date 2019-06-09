import tensorflow as tf
print('TensorFlow:{}'.format(tf.__version__))
import numpy as np
print('NumPy:{}'.format(np.__version__))
import matplotlib.pyplot as plt
import matplotlib
print('Matplotlib:{}'.format(matplotlib.__version__))


x_data = np.random.randn(5, 10)
w_data = np.random.randn(10, 1)

print('x_data: {}'.format(x_data.shape))
print('w_data: {}'.format(w_data))

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=(5, 10))
    w = tf.placeholder(tf.float32, shape=(10, 1))
    b = tf.fill((5, 1), -1.)
    xw = tf.matmul(x, w)
    xwb = xw + b
    s = tf.reduce_max(xwb)
    with tf.Session() as tfs:
        outs = tfs.run(s, feed_dict={x: x_data, w: w_data})
print('out = {}'.format(outs))

