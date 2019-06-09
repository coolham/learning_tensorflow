import tensorflow as tf
print('TensorFlow:{}'.format(tf.__version__))
import numpy as np
print('NumPy:{}'.format(np.__version__))
import matplotlib.pyplot as plt
import matplotlib
print('Matplotlib:{}'.format(matplotlib.__version__))



x_data = np.random.randn(2000, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2

noise = np.random.randn(1, 2000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise

print('y_data shape: {}'.format(y_data.shape))

NUM_STEPS = 10

g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)

    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='w')
        b = tf.Variable(0, dtype=tf.float32, name='b')
        y_pred = tf.matmul(w, tf.transpose(x)) + b

    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as tfs:
        tfs.run(init)
        for step in range(NUM_STEPS):
            train_val = tfs.run(train, {x: x_data, y_true: y_data})
            if (step % 5 ) == 0:
                print('{0} {1}'.format(step, tfs.run([w, b])))
                wb_.append(tfs.run([w, b]))

        w_val, b_val = tfs.run([w, b])
        print('w: {}, b: {}'.format(w_val, b_val))


plt.figure(figsize=(13, 7))
plt.scatter(x_data[:, 0], y_data, marker='o', c=y_data)
plt.title('Original')
x_plot = x_data
y_plot =  x_plot * w_val + b_val

plt.plot(x_plot, y_plot, 'r-', label='Trained Model')
plt.legend()
plt.show()

