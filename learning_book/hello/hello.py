import tensorflow as tf
print('TensorFlow:{}'.format(tf.__version__))
import numpy as np
print('NumPy:{}'.format(np.__version__))
import matplotlib.pyplot as plt
import matplotlib
print('Matplotlib:{}'.format(matplotlib.__version__))

h = tf.constant('Hello')
w = tf.constant('World!')
hw = h + w

with tf.Session() as tfs:
    ans = tfs.run(hw)

print(ans)




