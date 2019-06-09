import numpy as np
import tensorflow as tf


import pysnooper

@pysnooper.snoop()
def test():
    elems = np.array(['T', 'e', 'n', 's', 'o', 'r', ' ', 'F', 'l', 'o', 'w'])
    scan_sum = tf.scan(lambda a, x: a + x, elems)

    sess = tf.InteractiveSession()
    r = sess.run(scan_sum)
    print(r)


test()
