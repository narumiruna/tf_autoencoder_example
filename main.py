import tensorflow as tf

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# load iris data
iris = load_iris()
x, y = iris.data, iris.target


# build autoencoder
# z -- w --> code -- w^T --> z'
z_in = tf.placeholder(dtype=tf.float32, shape=[None, 4])
w = tf.Variable(tf.random_normal([4, 2]))
code = z_in @ w
z_out = code @ tf.transpose(w)

# define l2 loss
loss = tf.nn.l2_loss(z_in - z_out)

# choose optimization algorithm
minimizer = tf.train.AdagradOptimizer(0.03).minimize(loss)

# train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10**4):
        sess.run(minimizer, feed_dict={z_in: x})

        if i % 100 == 0:
            print('#{}, loss = {}'.format(i, sess.run(loss, feed_dict={z_in: x})))

    x_code = sess.run(code, feed_dict={z_in: x})


# plot
plt.scatter(x_code[:, 0], x_code[:, 1], c=y)
plt.show()
